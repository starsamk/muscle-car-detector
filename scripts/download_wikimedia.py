"""Download licensed training images from Wikimedia Commons categories."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final, Iterator

from dataset_config import (
    DEFAULT_TAXONOMY_PATH,
    DatasetConfigError,
    DatasetProfile,
    TaxonomyClass,
    load_profile,
    load_taxonomy,
    select_taxonomy_classes,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
API_URL: Final[str] = "https://commons.wikimedia.org/w/api.php"
USER_AGENT: Final[str] = (
    "CarSpotterAI/0.1 "
    "(https://github.com/starsamk/muscle-car-detector; "
    "open-source dataset research)"
)
ALLOWED_MIME_TYPES: Final[set[str]] = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_LICENSE_MARKERS: Final[tuple[str, ...]] = (
    "cc by",
    "cc-by",
    "cc0",
    "public domain",
    "pd-",
)
HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


class WikimediaError(RuntimeError):
    """Raised when the Wikimedia API or a download cannot be processed."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download Wikimedia images declared in the taxonomy."
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Path to the taxonomy JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/raw"),
        help="Directory receiving images and the JSONL manifest.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=80,
        help="Maximum number of images downloaded from each category.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        help="Stop collecting a class after this many manifest records exist.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=1600,
        help="Requested Wikimedia thumbnail width.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay in seconds between downloaded files.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSONL manifest whose source titles, pages, and SHA-256 hashes must "
            "not be downloaded; repeat for multiple manifests."
        ),
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--profile",
        type=Path,
        help="Optional dataset profile limiting the selected classes.",
    )
    selection_group.add_argument(
        "--class-slug",
        action="append",
        default=[],
        help="Download only selected class slugs; repeat for multiple classes.",
    )
    return parser.parse_args()


def load_manifest_exclusions(
    paths: Iterable[Path],
) -> tuple[set[str], set[str], set[str]]:
    """Load source identities that must not enter a new collection.

    Args:
        paths: Existing JSONL manifests used as exclusion references.

    Returns:
        Source titles, source pages, and SHA-256 hashes to exclude.

    Raises:
        WikimediaError: If an exclusion manifest is missing or malformed.
    """
    source_titles: set[str] = set()
    source_pages: set[str] = set()
    sha256_hashes: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise WikimediaError(f"Exclusion manifest not found: '{path}'.")
        try:
            lines: list[str] = path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                record: object = json.loads(line)
                if not isinstance(record, dict):
                    raise WikimediaError(
                        f"Exclusion manifest line {line_number} in '{path}' "
                        "is not an object."
                    )
                source_title: str = str(record.get("source_title", "")).strip()
                source_page: str = str(record.get("source_page", "")).strip()
                sha256_hash: str = str(record.get("sha256", "")).strip()
                if source_title:
                    source_titles.add(source_title)
                if source_page:
                    source_pages.add(source_page)
                if sha256_hash:
                    sha256_hashes.add(sha256_hash)
        except (OSError, json.JSONDecodeError) as error:
            raise WikimediaError(
                f"Unable to read exclusion manifest '{path}'."
            ) from error
    return source_titles, source_pages, sha256_hashes


def api_request(parameters: dict[str, str | int]) -> dict[str, Any]:
    """Send a read-only request to the Wikimedia Commons API.

    Args:
        parameters: MediaWiki API query parameters.

    Returns:
        Parsed API response.

    Raises:
        WikimediaError: If the request or response decoding fails.
    """
    query: dict[str, str | int] = {
        "format": "json",
        "formatversion": 2,
        **parameters,
    }
    url: str = f"{API_URL}?{urllib.parse.urlencode(query)}"
    request: urllib.request.Request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    payload: object | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise WikimediaError(f"Wikimedia API request failed: {url}") from error
            retry_after: str = error.headers.get("Retry-After", "")
            server_delay: float = float(retry_after) if retry_after.isdigit() else 0.0
            wait_seconds: float = max(server_delay, float(5 * 2**attempt))
            LOGGER.warning(
                "Wikimedia rate limit reached; retrying in %.1f seconds",
                wait_seconds,
            )
            time.sleep(wait_seconds)
        except (OSError, json.JSONDecodeError) as error:
            raise WikimediaError(f"Wikimedia API request failed: {url}") from error
    if payload is None:
        raise WikimediaError(f"Wikimedia API returned no payload: {url}")
    if not isinstance(payload, dict):
        raise WikimediaError("Wikimedia returned an invalid JSON response.")
    if "error" in payload:
        raise WikimediaError(f"Wikimedia API error: {payload['error']}")
    return payload


def iter_category_files(
    category: str, limit: int, thumbnail_width: int
) -> Iterator[dict[str, Any]]:
    """Yield file metadata from a Commons category.

    Args:
        category: Commons category name without the ``Category:`` prefix.
        limit: Maximum number of files yielded.
        thumbnail_width: Requested thumbnail width in pixels.

    Yields:
        API page dictionaries containing image information.
    """
    continuation: str | None = None
    yielded_count: int = 0
    while yielded_count < limit:
        parameters: dict[str, str | int] = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": min(50, limit - yielded_count),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": thumbnail_width,
        }
        if continuation is not None:
            parameters["gcmcontinue"] = continuation
        payload: dict[str, Any] = api_request(parameters)
        query: object = payload.get("query", {})
        pages: object = query.get("pages", []) if isinstance(query, dict) else []
        if not isinstance(pages, list):
            raise WikimediaError(f"Unexpected response for category '{category}'.")
        for page in pages:
            if isinstance(page, dict):
                yielded_count += 1
                yield page
        continuation_data: object = payload.get("continue")
        if not isinstance(continuation_data, dict):
            break
        continuation_value: object = continuation_data.get("gcmcontinue")
        if not isinstance(continuation_value, str):
            break
        continuation = continuation_value


def clean_metadata(value: object) -> str:
    """Convert Wikimedia extended metadata to readable plain text.

    Args:
        value: Raw metadata value or metadata mapping.

    Returns:
        Sanitized text.
    """
    if isinstance(value, dict):
        value = value.get("value", "")
    text_value: str = str(value) if value is not None else ""
    return html.unescape(HTML_TAG_PATTERN.sub("", text_value)).strip()


def is_allowed_license(license_name: str) -> bool:
    """Return whether a Commons license is accepted by this project.

    Args:
        license_name: Human-readable license name.

    Returns:
        ``True`` for permissive Creative Commons or public-domain licenses.
    """
    normalized_license: str = license_name.casefold()
    return any(marker in normalized_license for marker in ALLOWED_LICENSE_MARKERS)


def download_file(url: str, destination: Path) -> str:
    """Download one image atomically and return its SHA-256 digest.

    Args:
        url: Source image URL.
        destination: Final local image path.

    Returns:
        Hexadecimal SHA-256 digest.

    Raises:
        WikimediaError: If the image cannot be downloaded.
    """
    temporary_path: Path = destination.with_suffix(destination.suffix + ".part")
    request: urllib.request.Request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(6):
        digest: Any = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary_path.open("wb") as output_file:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        output_file.write(chunk)
            temporary_path.replace(destination)
            return digest.hexdigest()
        except urllib.error.HTTPError as error:
            temporary_path.unlink(missing_ok=True)
            if error.code != 429 or attempt == 5:
                raise WikimediaError(f"Unable to download '{url}'.") from error
            retry_after: str = error.headers.get("Retry-After", "")
            server_delay: float = float(retry_after) if retry_after.isdigit() else 0.0
            wait_seconds: float = max(server_delay, float(10 * 2**attempt))
            LOGGER.warning(
                "Image download rate limit reached; retrying in %.1f seconds",
                wait_seconds,
            )
            time.sleep(wait_seconds)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise WikimediaError(f"Unable to download '{url}'.") from error
    raise WikimediaError(f"Unable to download '{url}'.")


def extension_for_mime(mime_type: str) -> str:
    """Map a supported image MIME type to a file extension.

    Args:
        mime_type: Wikimedia MIME type.

    Returns:
        Lower-case filename extension including the leading dot.
    """
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[mime_type]


def process_page(
    page: dict[str, Any],
    class_definition: TaxonomyClass,
    category: str,
    output_directory: Path,
) -> dict[str, Any] | None:
    """Validate and download one API page.

    Args:
        page: Wikimedia API page object.
        class_definition: Taxonomy class assigned to the category.
        category: Source Wikimedia category.
        output_directory: Dataset root directory.

    Returns:
        Manifest record, or ``None`` when the file should be skipped.
    """
    image_info_list: object = page.get("imageinfo")
    if not isinstance(image_info_list, list) or not image_info_list:
        return None
    image_info: object = image_info_list[0]
    if not isinstance(image_info, dict):
        return None
    mime_type: str = str(image_info.get("mime", ""))
    if mime_type not in ALLOWED_MIME_TYPES:
        return None
    metadata: object = image_info.get("extmetadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    license_name: str = clean_metadata(metadata.get("LicenseShortName"))
    if not is_allowed_license(license_name):
        LOGGER.warning("Skipping unsupported license '%s'", license_name)
        return None

    source_url: str = str(image_info.get("thumburl") or image_info.get("url") or "")
    if not source_url:
        return None
    page_id: str = str(page.get("pageid", "unknown"))
    title: str = str(page.get("title", "untitled"))
    filename_key: str = hashlib.sha256(f"{page_id}:{title}".encode()).hexdigest()[:20]
    class_slug: str = class_definition.slug
    class_directory: Path = output_directory / "images" / class_slug
    class_directory.mkdir(parents=True, exist_ok=True)
    destination: Path = class_directory / (filename_key + extension_for_mime(mime_type))
    if destination.exists():
        LOGGER.info("Already downloaded: %s", destination)
        file_digest: str = hashlib.sha256(destination.read_bytes()).hexdigest()
    else:
        file_digest = download_file(source_url, destination)

    return {
        "class_slug": class_slug,
        "display_name": class_definition.display_name,
        "make": class_definition.make,
        "model": class_definition.model,
        "generation": class_definition.generation,
        "body_style": class_definition.body_style,
        "year_start": class_definition.year_start,
        "year_end": class_definition.year_end,
        "local_path": str(destination),
        "sha256": file_digest,
        "source": "wikimedia_commons",
        "source_category": category,
        "source_title": title,
        "source_page": str(image_info.get("descriptionurl", "")),
        "author": clean_metadata(metadata.get("Artist")),
        "credit": clean_metadata(metadata.get("Credit")),
        "license": license_name,
        "license_url": clean_metadata(metadata.get("LicenseUrl")),
    }


def main() -> None:
    """Download configured Wikimedia categories and write a JSONL manifest."""
    arguments: argparse.Namespace = parse_arguments()
    if arguments.limit_per_category <= 0:
        raise WikimediaError("--limit-per-category must be greater than zero.")
    if arguments.target_per_class is not None and arguments.target_per_class <= 0:
        raise WikimediaError("--target-per-class must be greater than zero.")
    taxonomy: dict[str, TaxonomyClass] = load_taxonomy(arguments.taxonomy)
    profile: DatasetProfile | None = (
        load_profile(arguments.profile, taxonomy)
        if arguments.profile is not None
        else None
    )
    classes: list[TaxonomyClass] = select_taxonomy_classes(
        taxonomy, profile, arguments.class_slug
    )
    output_directory: Path = arguments.output
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = output_directory / "manifest.jsonl"
    exclusion_manifests: list[Path] = list(arguments.exclude_manifest)
    if manifest_path.exists():
        exclusion_manifests.append(manifest_path)
    (
        excluded_source_titles,
        excluded_source_pages,
        excluded_sha256_hashes,
    ) = load_manifest_exclusions(exclusion_manifests)
    existing_records: set[tuple[str, str]] = set()
    class_counts: Counter[str] = Counter()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            record: object = json.loads(line)
            if isinstance(record, dict):
                class_slug: str = str(record.get("class_slug", ""))
                source_title: str = str(record.get("source_title", ""))
                existing_records.add((class_slug, source_title))
                class_counts[class_slug] += 1

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for class_definition in classes:
            class_slug: str = class_definition.slug
            if (
                arguments.target_per_class is not None
                and class_counts[class_slug] >= arguments.target_per_class
            ):
                LOGGER.info(
                    "Target already reached for %s: %d records",
                    class_slug,
                    class_counts[class_slug],
                )
                continue
            for category in class_definition.wikimedia_categories:
                LOGGER.info("Reading Category:%s for %s", category, class_slug)
                for page in iter_category_files(
                    category,
                    arguments.limit_per_category,
                    arguments.thumbnail_width,
                ):
                    if (
                        arguments.target_per_class is not None
                        and class_counts[class_slug] >= arguments.target_per_class
                    ):
                        break
                    source_title: str = str(page.get("title", ""))
                    if (
                        class_slug,
                        source_title,
                    ) in existing_records or source_title in excluded_source_titles:
                        continue
                    try:
                        record = process_page(
                            page, class_definition, category, output_directory
                        )
                    except WikimediaError:
                        LOGGER.exception("Failed to process %s", source_title)
                        continue
                    if record is None:
                        continue
                    source_page: str = str(record.get("source_page", "")).strip()
                    sha256_hash: str = str(record.get("sha256", "")).strip()
                    if (
                        source_page in excluded_source_pages
                        or sha256_hash in excluded_sha256_hashes
                    ):
                        local_path: Path = Path(str(record.get("local_path", "")))
                        local_path.unlink(missing_ok=True)
                        LOGGER.info("Skipping excluded source: %s", source_title)
                        continue
                    manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest_file.flush()
                    existing_records.add((class_slug, source_title))
                    excluded_source_titles.add(source_title)
                    if source_page:
                        excluded_source_pages.add(source_page)
                    if sha256_hash:
                        excluded_sha256_hashes.add(sha256_hash)
                    class_counts[class_slug] += 1
                    time.sleep(max(0.0, arguments.delay))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (DatasetConfigError, WikimediaError):
        LOGGER.exception("Wikimedia dataset download failed")
        raise SystemExit(1)
