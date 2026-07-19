import datetime
import io
import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

import scraper


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class XquikSearchTests(unittest.TestCase):
    def test_maps_xquik_search_response(self):
        response = JsonResponse(json.dumps({
            "tweets": [
                {
                    "id": "1234567890123456789",
                    "text": "Acme launch sentiment",
                    "createdAt": "2026-07-16T12:00:00Z",
                    "likeCount": 17,
                },
                {
                    "id": "9876543210987654321",
                    "text": "Acme stale sentiment",
                    "createdAt": "2026-07-01T12:00:00Z",
                    "likeCount": 4,
                }
            ]
        }).encode())
        fixed_now = datetime.datetime(
            2026,
            7,
            17,
            12,
            tzinfo=datetime.timezone.utc,
        )

        with patch.dict(
            "os.environ",
            {
                "XQUIK_API_KEY": "test-key",
                "XQUIK_API_BASE_URL": "https://example.test/api/v1/",
            },
            clear=True,
        ), patch(
            "scraper.datetime.datetime",
        ) as date_time, patch(
            "scraper.urlopen",
            return_value=response,
        ) as open_url:
            date_time.now.return_value = fixed_now
            result = scraper.search_xquik_mentions("Acme", limit=10, days_back=7)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["source"], "xquik_api")
        self.assertEqual(result.iloc[0]["brand"], "Acme")
        self.assertEqual(result.iloc[0]["score"], 17)
        self.assertEqual(
            result.iloc[0]["url"],
            "https://x.com/i/status/1234567890123456789",
        )

        request = open_url.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(query["q"], ["Acme"])
        self.assertEqual(query["queryType"], ["Latest"])
        self.assertEqual(query["limit"], ["10"])
        self.assertEqual(query["sinceTime"], ["2026-07-10T12:00:00Z"])
        self.assertTrue(
            (result["date"] >= pd.Timestamp("2026-07-10T12:00:00")).all()
        )
        self.assertEqual(request.get_header("X-api-key"), "test-key")
        self.assertEqual(
            open_url.call_args.kwargs["timeout"],
            scraper.XQUIK_REQUEST_TIMEOUT_SECONDS,
        )

    def test_skips_api_without_key(self):
        with patch.dict("os.environ", {}, clear=True), patch("scraper.urlopen") as open_url:
            result = scraper.search_xquik_mentions("Acme")

        self.assertIsNone(result)
        open_url.assert_not_called()

    def test_uses_api_after_missing_csv_override(self):
        expected = pd.DataFrame([{"text": "Acme mention"}])
        with patch(
            "scraper.load_xquik_csv_mentions",
            return_value=None,
        ) as load_csv, patch(
            "scraper.search_xquik_mentions",
            return_value=expected,
        ) as search_api:
            result = scraper.scrape_twitter("Acme", limit=20, days_back=14)

        self.assertIs(result, expected)
        load_csv.assert_called_once_with("Acme", limit=20, days_back=14)
        search_api.assert_called_once_with("Acme", limit=20, days_back=14)

    def test_csv_override_precedes_api(self):
        expected = pd.DataFrame([{"text": "Reviewed mention"}])
        with patch(
            "scraper.load_xquik_csv_mentions",
            return_value=expected,
        ), patch("scraper.search_xquik_mentions") as search_api:
            result = scraper.scrape_twitter("Acme")

        self.assertIs(result, expected)
        search_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
