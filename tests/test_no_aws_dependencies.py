import re
import subprocess
from pathlib import Path

import pytest

S3_HOST_PATTERN = re.compile(
    r"\bpyaging\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com(?:\.cn)?\b",
    re.IGNORECASE,
)
S3_PATH_STYLE_URL_PATTERN = re.compile(
    r"https?://s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com(?:\.cn)?/pyaging(?=[/?#]|$)",
    re.IGNORECASE,
)
S3_URI_PATTERN = re.compile(r"\bs3://pyaging(?=[/?#]|$)", re.IGNORECASE)
SHELL_TOKEN_SEPARATOR = r"(?:[^\S\r\n]+|[^\S\r\n]*\\\r?\n[^\S\r\n]*)"
AWSCLI_ENTRYPOINT = (
    rf"(?:aws|python(?:3(?:\.\d+)?)?{SHELL_TOKEN_SEPARATOR}-m{SHELL_TOKEN_SEPARATOR}awscli|"
    rf"uvx{SHELL_TOKEN_SEPARATOR}awscli)"
)
AWS_GLOBAL_OPTION = rf"--?[a-z0-9][a-z0-9-]*(?:=[^\s\\]+|{SHELL_TOKEN_SEPARATOR}(?!-)(?!s3(?:api)?\b)[^\s\\]+)?"
S3_HIGH_LEVEL_COMMAND = r"(?:cp|ls|mb|mv|presign|rb|rm|sync|website)"
S3API_COMMAND = (
    r"(?:wait|(?:abort|complete|copy|create|delete|download|get|head|list|put|restore|select|upload|write)"
    r"(?:-[a-z0-9]+)+)"
)
AWS_S3_OPERATION = rf"(?:s3{SHELL_TOKEN_SEPARATOR}{S3_HIGH_LEVEL_COMMAND}|s3api{SHELL_TOKEN_SEPARATOR}{S3API_COMMAND})"
AWS_S3_COMMAND_PATTERN = re.compile(
    rf"(?<![a-z0-9_]){AWSCLI_ENTRYPOINT}"
    rf"(?:{SHELL_TOKEN_SEPARATOR}{AWS_GLOBAL_OPTION})*{SHELL_TOKEN_SEPARATOR}{AWS_S3_OPERATION}\b",
    re.IGNORECASE,
)
SDK_BUCKET_ARGUMENT_PATTERN = re.compile(
    r"[\"']?\b(?:bucket|bucket_?name)\b[\"']?\s*(?:=|:)\s*(?:aws\.String\s*\(\s*)?[\"']pyaging[\"']",
    re.IGNORECASE,
)
SDK_BUCKET_CALL_PATTERN = re.compile(
    r"\b(?:bucket|get_bucket|create_bucket)\s*\(\s*[\"']pyaging[\"']",
    re.IGNORECASE,
)
SDK_BUCKET_FIRST_POSITIONAL_PATTERN = re.compile(
    r"\b(?:download_file(?:obj)?|object(?:summary)?)\s*\(\s*[\"']pyaging[\"']\s*,",
    re.IGNORECASE,
)
SDK_BUCKET_SECOND_POSITIONAL_PATTERN = re.compile(
    r"\bupload_file(?:obj)?\s*\([^\r\n]{0,500}?,\s*[\"']pyaging[\"']\s*,",
    re.IGNORECASE,
)
ACTIVE_SUFFIXES = {
    ".bash",
    ".bat",
    ".cfg",
    ".css",
    ".csv",
    ".fish",
    ".gitignore",
    ".htm",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".md",
    ".mdc",
    ".ps1",
    ".py",
    ".pyi",
    ".rst",
    ".scss",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
ACTIVE_FILENAMES = {
    ".dockerignore",
    ".gitignore",
    "containerfile",
    "dockerfile",
    "gnumakefile",
    "makefile",
    "pipfile",
}
EXCLUDED_PREFIXES = (("docs", "superpowers"),)
EXCLUDED_FIXTURES = {
    Path("tests/test_no_aws_dependencies.py"),
    Path("tests/test_notebook_hosts.py"),
}
FORBIDDEN_PATTERNS = {
    "pyaging S3 hostname": S3_HOST_PATTERN,
    "pyaging path-style S3 URL": S3_PATH_STYLE_URL_PATTERN,
    "pyaging S3 URI": S3_URI_PATTERN,
    "AWS S3 command": AWS_S3_COMMAND_PATTERN,
    "pyaging SDK bucket argument": SDK_BUCKET_ARGUMENT_PATTERN,
    "pyaging SDK bucket call": SDK_BUCKET_CALL_PATTERN,
    "pyaging SDK first positional bucket": SDK_BUCKET_FIRST_POSITIONAL_PATTERN,
    "pyaging SDK second positional bucket": SDK_BUCKET_SECOND_POSITIONAL_PATTERN,
}


def _active_files(repository_root):
    tracked_files = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for tracked_file in tracked_files:
        if not tracked_file:
            continue
        relative_path = Path(tracked_file)
        if relative_path in EXCLUDED_FIXTURES:
            continue
        if any(relative_path.parts[: len(prefix)] == prefix for prefix in EXCLUDED_PREFIXES):
            continue
        filename = relative_path.name.casefold()
        is_named_source = filename in ACTIVE_FILENAMES or filename.startswith(("containerfile.", "dockerfile."))
        if relative_path.suffix.casefold() in ACTIVE_SUFFIXES or is_named_source:
            yield repository_root / relative_path


def _forbidden_references(repository_root):
    offenders = []
    for path in _active_files(repository_root):
        content = path.read_text(encoding="utf-8")
        for description, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(content):
                relative_path = path.relative_to(repository_root).as_posix()
                offenders.append(f"{relative_path}: {description}")
    return sorted(offenders)


@pytest.mark.parametrize(
    "hostname",
    [
        "pyaging.s3.amazonaws.com",
        "pyaging.s3.us-east-1.amazonaws.com",
        "pyaging.s3-us-east-1.amazonaws.com",
        "pyaging.s3.dualstack.us-east-1.amazonaws.com",
        "pyaging.s3.cn-north-1.amazonaws.com.cn",
    ],
)
def test_s3_host_detector_catches_aws_hostname_variants(hostname):
    assert S3_HOST_PATTERN.search(hostname)


@pytest.mark.parametrize(
    "reference",
    [
        "https://s3.amazonaws.com/pyaging/clock.pt",
        "http://s3.amazonaws.com/pyaging/clock.pt",
        "https://s3.us-east-1.amazonaws.com/pyaging/clock.pt",
        "https://s3-us-east-1.amazonaws.com/pyaging/clock.pt",
        "https://s3.dualstack.us-east-1.amazonaws.com/pyaging/clock.pt",
        "https://s3.cn-north-1.amazonaws.com.cn/pyaging/clock.pt",
        "s3://pyaging/clock.pt",
    ],
)
def test_s3_detector_catches_path_style_urls_and_bucket_uris(reference):
    assert any(pattern.search(reference) for pattern in FORBIDDEN_PATTERNS.values())


@pytest.mark.parametrize(
    "reference",
    [
        "aws s3 cp source destination",
        "AWS  S3 sync source destination",
        "aws --profile release s3 sync source destination",
        "aws --region us-east-1 s3api list-objects --bucket pyaging",
        "python -m awscli s3 sync source destination",
        "uvx awscli s3api list-objects --bucket pyaging",
        "@aws s3 cp source destination",
        "-aws s3 sync source destination",
        'run: "aws s3 sync source destination',
        "AWS_PROFILE=release aws s3 cp source destination",
        "sudo aws s3 sync source destination",
        "bash -c 'aws s3 cp source destination'",
        "(aws s3 sync source destination)",
        "aws \\\n  --profile release \\\n  s3 sync source destination",
    ],
)
def test_aws_s3_command_detector_catches_whitespace_and_case_variants(reference):
    assert AWS_S3_COMMAND_PATTERN.search(reference)


@pytest.mark.parametrize(
    "reference",
    [
        "AWS is no longer used; all S3 dependencies were removed",
        "AWS S3 dependencies were removed",
    ],
)
def test_aws_s3_command_detector_does_not_match_prose(reference):
    assert not AWS_S3_COMMAND_PATTERN.search(reference)


@pytest.mark.parametrize(
    "reference",
    [
        'client.get_object(Bucket="pyaging", Key=key)',
        'request = {"Bucket": "pyaging"}',
        'bucket_name = "pyaging"',
        'bucket: aws.String("pyaging")',
        's3.Bucket("pyaging")',
        'builder.bucket("pyaging")',
    ],
)
def test_sdk_bucket_detector_catches_named_arguments_and_calls(reference):
    assert any(pattern.search(reference) for pattern in FORBIDDEN_PATTERNS.values())


@pytest.mark.parametrize(
    "reference",
    [
        's3.download_file("pyaging", key, filename)',
        's3.download_fileobj("pyaging", key, fileobj)',
        's3.upload_file(filename, "pyaging", key)',
        's3.upload_fileobj(fileobj, "pyaging", key)',
        's3.upload_file(os.path.join("a", "b"), "pyaging", key)',
        'boto3.resource("s3").Object("pyaging", key)',
        'boto3.resource("s3").ObjectSummary("pyaging", key)',
    ],
)
def test_sdk_bucket_detector_catches_positional_boto3_calls(reference):
    assert any(pattern.search(reference) for pattern in FORBIDDEN_PATTERNS.values())


def test_active_file_discovery_uses_tracked_text_and_config_sources(tmp_path):
    included = {
        "pyaging/module.py",
        "clocks/notebooks/clock.ipynb",
        "docs/source/page.rst",
        ".github/workflows/test.yml",
        "Makefile",
        "docs/Makefile",
        "pyproject.toml",
        "scripts/release.sh",
        "Dockerfile",
        "Containerfile",
        ".github/actions/setup/action.yml",
        "tutorials/example.ipynb",
        "clocks/publish.py",
        "tests/test_release_configuration.py",
    }
    excluded = {
        "docs/superpowers/archive.md",
        "tests/test_no_aws_dependencies.py",
        "tests/test_notebook_hosts.py",
        "docs/source/image.png",
    }
    for relative_path in included | excluded:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    discovered = {path.relative_to(tmp_path).as_posix() for path in _active_files(tmp_path)}

    assert included <= discovered
    assert discovered.isdisjoint(excluded)


def test_active_sources_do_not_depend_on_aws_data_hosting():
    repository_root = Path(__file__).resolve().parents[1]

    assert _forbidden_references(repository_root) == []
