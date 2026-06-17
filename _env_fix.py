"""
Windows import-order workaround. IMPORT THIS FIRST, before `config` or torch.

On this machine a filesystem-transaction filter (antivirus / Controlled Folder
Access) makes Python's package-directory scan of the project root fail with
`OSError: [WinError 6714] ... not a valid handle to a transaction object`
*if* an import-time CUDA call (`torch.cuda.is_available()` in config.py) runs
before pyarrow/datasets is first imported. Importing pyarrow & datasets up
front — while the filter is still quiescent — makes their directory scans
happen before anything arms the bad state, after which everything works
(verified: torch.cuda.is_available() == True, full pipeline runs).

This changes import ORDER only. It does not alter any experiment behaviour,
library versions, or detector logic. Pure environment shim; safe to remove on
a machine without the offending security filter.
"""
import pyarrow  # noqa: F401  (must precede the import-time CUDA init)
import datasets  # noqa: F401
