# Tests

This folder contains all the integration tests for dyntapy. 
`test_dyntapy.py` hosts all functions that are tested during a pipeline run.

`pytest` will find all files of the form test_\*.py and collect all functions of the form test\* within them for testing.

```shell
pytest /path/to/repo 
```

will trigger the collection and execution of all tests that are not explicitly skipped.

