from __future__ import annotations


class PipelineError(Exception):
    pass


class ConfigurationError(PipelineError):
    pass


class InputParseError(PipelineError):
    pass


class MetadataError(PipelineError):
    pass


class IdentifierRecoveryError(PipelineError):
    pass


class ResolutionError(PipelineError):
    pass


class DownloadError(PipelineError):
    pass


class PDFValidationError(PipelineError):
    pass


class ExtractionError(PipelineError):
    pass


class CaptioningError(PipelineError):
    pass


class ResumeError(PipelineError):
    pass


class ManifestStoreError(PipelineError):
    pass