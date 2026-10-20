import hashlib
import json
import logging
import re
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"

# Bumped whenever parsing/extraction logic changes materially, so
# downstream RAG results can be traced back to the exact parser
# version that produced a given document's metadata.
PARSER_VERSION = "1.0"


class DocumentProcessor:
    """
    Processes financial PDF documents using Docling.

    Outputs:
        - Markdown representation of the document (with page-break
          markers preserved for downstream page-level attribution)
        - Individual tables as CSV files
        - Table metadata, including a heuristically-extracted
          table_heading and context line for each table (see
          _build_table_context_index). This is NOT true table
          classification — it only captures a heading/caption if
          one exists directly above the table, on the same PDF
          page, in the source document.
        - Document metadata (including a source file hash for dedup
          and the parser_version that produced it)

    Output layout:
        <output_root>/<company_slug>/<financial_year_slug>/
            <stem>.md
            page_segments.json
            table_metadata.json
            document_metadata.json
            tables/*.csv

    Company, report name, and financial year are always parsed from
    the source filename (see _parse_filename) — there is no manual
    override — so this class works across any company/document
    without per-file hardcoding.
    """

    # Expected filename: "Company_ReportName_YYYY-YY" or
    # "Company_ReportName_YYYY-YYYY", e.g.
    # "Reliance_AnnualReport_2024-2025". ReportName must be
    # PascalCase (no spaces/underscores within it) so it can be
    # split back into words with _split_pascal_case. The year range
    # is deliberately restricted to exactly these two shapes (not
    # e.g. a stray 3-digit year) — anything else fails to match and
    # _parse_filename raises rather than guessing.
    FILENAME_PATTERN = re.compile(
        r"^(?P<company>[^_]+)_(?P<report>[^_]+)_"
        r"(?P<year_range>\d{4}-(?:\d{2}|\d{4}))$"
    )


    def __init__(self):
        """
        Initialize the Docling processing pipeline.
        """

        pipeline_options = PdfPipelineOptions()

        # The Reliance annual report contains machine-readable text,
        # so OCR is disabled.
        pipeline_options.do_ocr = False

        # Enable table structure recognition.
        pipeline_options.do_table_structure = True

        # FAST was selected after testing representative pages.
        # ACCURATE was measured at 10+ minutes per page, which is
        # not viable across multi-hundred-page annual reports or
        # multiple documents.
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.FAST
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    @staticmethod
    def _slugify_company(company: str) -> str:
        """
        Company token, lowercased, alphanumeric only.
        'Reliance' -> 'reliance', 'TCS' -> 'tcs', 'HDFCBank' -> 'hdfcbank'.
        """
        return "".join(ch for ch in company.strip().lower() if ch.isalnum())

    @staticmethod
    def _format_financial_year(financial_year: str) -> str:
        """
        Closing year of the FY, as four digits: '2024-25' -> '2025',
        '2024-2025' -> '2025'.

        Strict by design — the filename convention is deliberate, so
        anything that isn't exactly 'YYYY-YY' or 'YYYY-YYYY' (both
        halves all-digit, start exactly 4 digits) raises ValueError
        rather than guessing a year from malformed input.
        """
        error = ValueError(
            f"financial_year '{financial_year}' is not in the "
            f"required 'YYYY-YY' or 'YYYY-YYYY' format."
        )

        if "-" not in financial_year:
            raise error

        start, end = financial_year.split("-", 1)
        start = start.strip()
        end = end.strip()

        if not (start.isdigit() and end.isdigit() and len(start) == 4):
            raise error

        if len(end) == 4:
            return end
        if len(end) == 2:
            return start[:2] + end

        raise error

    @staticmethod
    def _split_pascal_case(text: str) -> str:
        """
        'AnnualReport' -> 'Annual Report'
        'GSTReport' -> 'GST Report' (acronyms kept together)
        """
        spaced = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            " ",
            text,
        )
        return spaced.strip()

    @classmethod
    def _parse_filename(cls, pdf_path: Path) -> dict:
        """
        Parse company, document_name, and financial_year out of a
        filename following the 'Company_ReportName_YYYY-YY(YY)'
        convention, e.g. 'Reliance_AnnualReport_2024-2025'.

        Raises ValueError with a clear message if the filename
        doesn't match this convention — rename the file rather than
        relying on any fallback guessing.
        """

        stem = pdf_path.stem
        match = cls.FILENAME_PATTERN.match(stem)

        if not match:
            raise ValueError(
                f"Filename '{pdf_path.name}' does not match the "
                f"expected 'Company_ReportName_YYYY-YY' or "
                f"'Company_ReportName_YYYY-YYYY' pattern (e.g. "
                f"'Reliance_AnnualReport_2024-2025'). Rename the "
                f"file to match this convention."
            )

        company = match.group("company")
        document_name = cls._split_pascal_case(match.group("report"))
        financial_year = match.group("year_range")

        return {
            "company": company,
            "document_name": document_name,
            "financial_year": financial_year,
        }

    def _build_table_context_index(self, docling_doc) -> dict:
        """
        Walk the document in reading order and, for each table,
        record the nearest preceding heading on the SAME PDF page
        (candidate "table_heading", e.g. "Balance Sheet") and the
        nearest preceding short text line on the same page
        (candidate "context", e.g. "As at 31st March, 2025").

        Page-scoped by design: heading/context state resets whenever
        the page number changes, so a heading from an earlier page
        can never be attached to a table on a later page.

        This is a heuristic, not semantic table classification: it
        only works when the source PDF has an explicit heading or
        caption line directly above the table, on the same page.
        Tables with nothing nearby on their own page will get
        table_heading=None / context=None and may need manual
        review, or a separate LLM-based classification pass over the
        table headers/content as a follow-up step.

        Returns a dict keyed by id(table_item) ->
        {"table_heading", "context"}. The docling_doc.iterate_items()
        API and DocItemLabel import path can vary across
        docling-core versions — if the import fails, this degrades
        to returning an empty index rather than crashing the whole
        run.
        """

        context_index: dict = {}

        try:
            from docling_core.types.doc import DocItemLabel
        except ImportError:
            logger.warning(
                "Could not import DocItemLabel from docling_core; "
                "table_heading/context extraction will be skipped "
                "for this run. Verify the import path against your "
                "installed docling-core version."
            )
            return context_index

        # A short line directly above a table (e.g. a date line) is
        # a much stronger "context" signal than a long paragraph, so
        # cap how long a text block can be to qualify as context.
        MAX_CONTEXT_CHARS = 120

        current_page = None
        current_heading = None
        recent_short_text = None

        for item, _level in docling_doc.iterate_items():

            prov = getattr(item, "prov", None)
            item_page = prov[0].page_no if prov else None

            # Page boundary: drop any heading/context carried over
            # from a previous page before looking at this item, so
            # it can never leak onto a table on a different page.
            # Items with no resolvable page (item_page is None) are
            # left as-is rather than forcing a reset on unrelated
            # structural items.
            if item_page is not None and item_page != current_page:
                current_page = item_page
                current_heading = None
                recent_short_text = None

            label = getattr(item, "label", None)

            is_table = (
                label == DocItemLabel.TABLE
                or type(item).__name__ == "TableItem"
            )

            if is_table:
                context_index[id(item)] = {
                    "table_heading": current_heading,
                    "context": recent_short_text,
                }
                # Reset so this context isn't reused for the next
                # table, whether on this page or a later one.
                recent_short_text = None
                continue

            text = getattr(item, "text", None)

            if not text:
                continue

            if label in (
                getattr(DocItemLabel, "SECTION_HEADER", None),
                getattr(DocItemLabel, "TITLE", None),
            ):
                current_heading = text.strip()
                recent_short_text = None
            elif len(text.strip()) <= MAX_CONTEXT_CHARS:
                recent_short_text = text.strip()

        return context_index

    def get_output_dir(
        self,
        pdf_path: str | Path,
        output_root: str | Path,
    ) -> Path:
        """
        Compute the output directory a given PDF would be processed
        into, without actually processing it — used by
        is_already_processed and available for callers that just
        need to know the destination path.
        """

        pdf_path = Path(pdf_path)
        output_root = Path(output_root)

        parsed = self._parse_filename(pdf_path)
        company_slug = self._slugify_company(parsed["company"])
        fy_slug = self._format_financial_year(parsed["financial_year"])

        return output_root / company_slug / fy_slug

    def is_already_processed(
        self,
        pdf_path: str | Path,
        output_root: str | Path,
    ) -> bool:
        """
        Return True if pdf_path has already been processed into
        output_root with the current PARSER_VERSION and the file
        hasn't changed since, so process_directory can skip it.

        Checks, in order:
          1. document_metadata.json exists at the expected output
             location for this filename.
          2. It's readable JSON.
          3. Its "parser_version" matches the current PARSER_VERSION
             — an older version means the extraction logic has
             since changed, so it's treated as NOT processed (needs
             reprocessing to pick up the improvements).
          4. Its "source_file_sha256" matches a fresh hash of
             pdf_path — a changed hash (e.g. a restated report
             re-saved under the same filename) means it's treated
             as NOT processed.

        Any failure to satisfy the above (missing file, corrupt
        JSON, missing fields, mismatched filename convention) is
        treated as "not processed" rather than raising, so this is
        always safe to call as a pre-check before process().
        """

        pdf_path = Path(pdf_path)
        output_root = Path(output_root)

        try:
            output_dir = self.get_output_dir(pdf_path, output_root)
        except ValueError:
            # Filename doesn't match the naming convention at all —
            # process() will raise the same error with a clearer
            # message, so just report "not processed" here.
            return False

        document_metadata_path = output_dir / "document_metadata.json"

        if not document_metadata_path.exists():
            return False

        try:
            with open(document_metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Could not read existing metadata at "
                f"{document_metadata_path} ({e}); treating "
                f"{pdf_path.name} as not processed."
            )
            return False

        if existing_metadata.get("parser_version") != PARSER_VERSION:
            logger.info(
                f"{pdf_path.name} was processed with parser_version="
                f"{existing_metadata.get('parser_version')!r}, "
                f"current is {PARSER_VERSION!r}; will reprocess."
            )
            return False

        if not pdf_path.exists():
            # Can't verify the hash without the source file — err
            # on the side of reprocessing rather than trusting stale
            # metadata blindly.
            return False

        current_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        if existing_metadata.get("source_file_sha256") != current_hash:
            logger.info(
                f"{pdf_path.name} content has changed since it was "
                f"last processed (hash mismatch); will reprocess."
            )
            return False

        return True

    def process(
        self,
        pdf_path: str | Path,
        output_root: str | Path,
    ):
        """
        Process a single PDF and save the structured outputs.

        Company, report name, and financial year are always parsed
        from pdf_path's filename (see _parse_filename) — expected
        as 'Company_ReportName_YYYY-YY(YY)', e.g.
        'Reliance_AnnualReport_2024-2025.pdf'. There is no manual
        override; rename the file if it doesn't fit the convention.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF.

        output_root:
            Base directory for processed output (e.g. ".../processed").
            The actual output goes to
            <output_root>/<company_slug>/<financial_year_slug>/.
        """

        pdf_path = Path(pdf_path)
        output_root = Path(output_root)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        parsed = self._parse_filename(pdf_path)
        company = parsed["company"]
        document_name = parsed["document_name"]
        financial_year = parsed["financial_year"]

        company_slug = self._slugify_company(company)
        fy_slug = self._format_financial_year(financial_year)

        output_dir = output_root / company_slug / fy_slug

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        tables_dir = output_dir / "tables"

        tables_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info(f"Processing: {pdf_path.name}")

        # Hash the source file so re-runs / restated reports can be
        # detected instead of silently overwriting prior output.
        file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        start_time = time.perf_counter()

        try:
            result = self.converter.convert(pdf_path)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            raise

        elapsed_time = time.perf_counter() - start_time

        docling_doc = result.document

        logger.info(
            f"Processing completed in "
            f"{elapsed_time / 60:.2f} minutes."
        )

        logger.info(
            f"Pages detected: {len(docling_doc.pages)}"
        )

        logger.info(
            f"Tables detected: {len(docling_doc.tables)}"
        )

        # --------------------------------------------------------
        # Save Markdown
        # --------------------------------------------------------
        # page_break_placeholder preserves page boundaries in the
        # markdown output, so a downstream chunker can assign an
        # exact page number to every chunk (needed for citation
        # accuracy in the RAG pipeline). Verify this kwarg name
        # against your installed docling version with
        # help(docling_doc.export_to_markdown) if it's ever renamed.

        markdown_text = docling_doc.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER
        )

        markdown_path = (
            output_dir /
            f"{pdf_path.stem}.md"
        )

        markdown_path.write_text(
            markdown_text,
            encoding="utf-8"
        )

        logger.info(
            f"Markdown saved: {markdown_path}"
        )

        # --------------------------------------------------------
        # Save page-tagged text segments
        # --------------------------------------------------------
        # Splitting on the placeholder gives one text segment per
        # page, in document order. This is what the chunker should
        # consume instead of the raw markdown file, so every chunk
        # can carry an accurate page number.

        page_segments = markdown_text.split(PAGE_BREAK_PLACEHOLDER)

        page_segments_data = [
            {
                "page": page_number,
                "text": segment.strip(),
            }
            for page_number, segment in enumerate(
                page_segments,
                start=1
            )
        ]

        if len(page_segments_data) != len(docling_doc.pages):
            logger.warning(f"Mismatch: {len(page_segments_data)} vs {len(docling_doc.pages)} "f"for {pdf_path.name}. Repairing segments...")
            
            while len(page_segments_data) < len(docling_doc.pages):
                page_segments_data.append({"page": len(page_segments_data) + 1, "text": ""})

            if len(page_segments_data) > len(docling_doc.pages):
                page_segments_data = page_segments_data[:len(docling_doc.pages)]


        page_segments_path = (
            output_dir /
            "page_segments.json"
        )

        with open(
            page_segments_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                page_segments_data,
                f,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            f"Page segments saved: {page_segments_path}"
        )

        # --------------------------------------------------------
        # Save tables
        # --------------------------------------------------------
        # company_slug / fy_slug were already computed above to
        # build output_dir, so they're reused here for table_id.

        table_context_index = self._build_table_context_index(
            docling_doc
        )

        table_metadata = []

        for table_number, table_item in enumerate(
            docling_doc.tables,
            start=1
        ):

            df = table_item.export_to_dataframe()

            if table_item.prov:
                page_number = table_item.prov[0].page_no
            else:
                page_number = None

            if page_number is not None:
                filename = (
                    f"page_{page_number:03d}_"
                    f"table_{table_number:03d}.csv"
                )
                table_id = (
                    f"{company_slug}_{fy_slug}_"
                    f"p{page_number:03d}_t{table_number:03d}"
                )
            else:
                filename = (
                    f"table_{table_number:03d}.csv"
                )
                table_id = (
                    f"{company_slug}_{fy_slug}_t{table_number:03d}"
                )

            table_path = tables_dir / filename

            df.to_csv(
                table_path,
                index=False,
                encoding="utf-8-sig"
            )

            context = table_context_index.get(
                id(table_item),
                {"table_heading": None, "context": None}
            )

            table_metadata.append(
                {
                    "table_id": table_id,
                    "table_number": table_number,
                    "page": page_number,
                    "filename": filename,
                    "company": company,
                    "document": document_name,
                    "financial_year": financial_year,
                    "table_heading": context["table_heading"],
                    "context": context["context"],
                    "rows": len(df),
                    "columns": len(df.columns),
                }
            )

        # --------------------------------------------------------
        # Save table metadata
        # --------------------------------------------------------

        table_metadata_path = (
            output_dir /
            "table_metadata.json"
        )

        with open(
            table_metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                table_metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        # --------------------------------------------------------
        # Save document metadata
        # --------------------------------------------------------

        document_metadata = {
            "company": company,
            "document": document_name,
            "financial_year": financial_year,
            "source_file": pdf_path.name,
            "source_file_sha256": file_hash,
            "number_of_pages": len(
                docling_doc.pages
            ),
            "number_of_tables": len(
                docling_doc.tables
            ),
            "processing_time_seconds": round(
                elapsed_time,
                2
            ),
            "processing_time_minutes": round(
                elapsed_time / 60,
                2
            ),
            "parser": "Docling",
            "parser_version": PARSER_VERSION,
            "ocr": False,
            "table_structure_detection": True,
            "tableformer_mode": "FAST",
        }

        document_metadata_path = (
            output_dir /
            "document_metadata.json"
        )

        with open(
            document_metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                document_metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            f"Table metadata saved: "
            f"{table_metadata_path}"
        )

        logger.info(
            f"Document metadata saved: "
            f"{document_metadata_path}"
        )

        return {
            "document": docling_doc,
            "markdown_path": markdown_path,
            "page_segments_path": page_segments_path,
            "tables_dir": tables_dir,
            "table_metadata_path": table_metadata_path,
            "document_metadata_path": document_metadata_path,
        }

    def process_directory(
        self,
        raw_dir: str | Path,
        output_root: str | Path,
        skip_processed: bool = True,
    ) -> dict:
        """
        Process every PDF in raw_dir into
        <output_root>/<company>/<year>/, per file, based purely on
        each filename — no per-company code needed.

        skip_processed (default True): before processing each file,
        check is_already_processed() and skip it if the output
        already exists, matches the current PARSER_VERSION, and the
        source file hasn't changed since. This is what saves the
        re-run time on a growing raw/ folder — only genuinely new or
        changed PDFs get (re)processed. Pass False to force
        reprocessing everything regardless.

        A failure on one file is logged and skipped rather than
        aborting the whole batch, since a single malformed filename
        or corrupt PDF shouldn't block the rest of the corpus.

        Returns {"succeeded": [...], "skipped": [...],
        "failed": [(path, error), ...]}.
        """

        raw_dir = Path(raw_dir)
        output_root = Path(output_root)

        # iterdir() + suffix check (rather than glob("*.pdf")) so
        # both ".pdf" and ".PDF" are picked up on case-sensitive
        # filesystems, without double-counting on filesystems (e.g.
        # Windows) where glob is already case-insensitive.
        pdf_paths = sorted(
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )

        if not pdf_paths:
            logger.warning(f"No PDFs found in {raw_dir}")

        succeeded = []
        skipped = []
        failed = []

        for pdf_path in pdf_paths:

            if skip_processed and self.is_already_processed(
                pdf_path, output_root
            ):
                logger.info(
                    f"Already processed, skipping: {pdf_path.name}"
                )
                skipped.append(pdf_path)
                continue

            try:
                self.process(
                    pdf_path=pdf_path,
                    output_root=output_root,
                )
                succeeded.append(pdf_path)
            except Exception as e:
                logger.error(
                    f"Skipping {pdf_path.name} due to error: {e}"
                )
                failed.append((pdf_path, e))

        logger.info(
            f"Batch complete: {len(succeeded)} processed, "
            f"{len(skipped)} already up to date, "
            f"{len(failed)} failed out of {len(pdf_paths)} PDF(s)."
        )

        if failed:
            for pdf_path, error in failed:
                logger.warning(f"  Failed: {pdf_path.name} — {error}")

        return {"succeeded": succeeded, "skipped": skipped, "failed": failed}


if __name__ == "__main__":

    # Every PDF in raw_dir is processed automatically based on its
    # filename ("Company_ReportName_YYYY-YYYY.pdf") -- nothing
    # hardcoded per-company or per-file here.
    raw_dir = Path(
        r"C:\Kiran\SRM\Semester 4\Project\financial-rag-india"
        r"\data\raw"
    )

    output_root = Path(
        r"C:\Kiran\SRM\Semester 4\Project\financial-rag-india"
        r"\data\processed"
    )

    processor = DocumentProcessor()

    processor.process_directory(
        raw_dir=raw_dir,
        output_root=output_root,
    )