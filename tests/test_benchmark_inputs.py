"""The remote inputs ``benchmarks/disco.py`` names exist in the public dataset, and a missing one is reported by name."""
import os
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import disco as bench


class _HeadRedirects(urllib.request.HTTPRedirectHandler):
    """Follows the hub's redirects with HEAD (urllib's own handler turns a redirected HEAD into a GET)."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return urllib.request.Request(newurl, method="HEAD")


@pytest.mark.parametrize("filename", bench.REMOTE_INPUTS)
def test_remote_input_exists(filename):
    """One HEAD request per input (followed through the hub's redirects as HEADs): HTTP 200 passes, any other status fails (a 404 is a file the benchmark names that
    the dataset does not hold). Skipped only when the host cannot be reached at all."""
    url = f"https://huggingface.co/datasets/{bench.HF_REPO}/resolve/main/{filename}"
    try:
        with urllib.request.build_opener(_HeadRedirects).open(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:        # no answer from the host: nothing to assert
        pytest.skip(f"huggingface.co cannot be reached: {e!r}")
    assert status == 200, f"HEAD {url} answered HTTP {status}"


def test_load_volume_names_the_missing_input(monkeypatch):
    """The library's not-found error becomes a FileNotFoundError that names the file and the dataset."""
    import huggingface_hub as hub
    from huggingface_hub.errors import EntryNotFoundError

    def not_found(**kw):
        raise EntryNotFoundError(f"404: {kw['filename']}")
    monkeypatch.setattr(hub, "hf_hub_download", not_found)
    with pytest.raises(FileNotFoundError) as e:
        bench.load_volume(None, None, None)
    assert bench.REMOTE_INPUTS[0] in str(e.value) and bench.HF_REPO in str(e.value)
