import datetime
import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

import scraper


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class XquikSearchTests(unittest.TestCase):
    def test_maps_current_xquik_search_response(self):
        response = JsonResponse(
            json.dumps(
                {
                    "tweets": [
                        {
                            "id": "1234567890123456789",
                            "text": "Acme launch sentiment",
                            "created": 1784203200,
                            "like_count": 17,
                        },
                        {
                            "id": "9876543210987654321",
                            "text": "Acme stale sentiment",
                            "created": 1782907200,
                            "like_count": 4,
                        },
                    ]
                }
            ).encode()
        )
        fixed_now = datetime.datetime(
            2026,
            7,
            17,
            12,
            tzinfo=datetime.timezone.utc,
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "XQUIK_API_KEY": "test-key",
                    "XQUIK_API_BASE_URL": "https://attacker.invalid/api/v1/",
                },
                clear=True,
            ),
            patch(
                "scraper.datetime.datetime",
            ) as date_time,
            patch(
                "scraper.urlopen",
                return_value=response,
            ) as open_url,
        ):
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
        self.assertEqual(
            f"{urlparse(request.full_url).scheme}://{urlparse(request.full_url).netloc}",
            "https://xquik.com",
        )
        self.assertTrue((result["date"] >= pd.Timestamp("2026-07-10T12:00:00")).all())
        self.assertEqual(request.get_header("X-api-key"), "test-key")
        self.assertEqual(
            request.get_header("Xquik-api-contract"),
            scraper.XQUIK_API_CONTRACT,
        )
        self.assertEqual(
            open_url.call_args.kwargs["timeout"],
            scraper.XQUIK_REQUEST_TIMEOUT_SECONDS,
        )

    def test_maps_legacy_xquik_timestamp(self):
        response = JsonResponse(
            json.dumps(
                {
                    "tweets": [
                        {
                            "id": "1234567890123456789",
                            "text": "Acme launch sentiment",
                            "createdAt": "2026-07-16T12:00:00Z",
                            "likeCount": 17,
                        }
                    ]
                }
            ).encode()
        )
        fixed_now = datetime.datetime(
            2026,
            7,
            17,
            12,
            tzinfo=datetime.timezone.utc,
        )

        with (
            patch.dict(
                "os.environ",
                {"XQUIK_API_KEY": "test-key"},
                clear=True,
            ),
            patch(
                "scraper.datetime.datetime",
            ) as date_time,
            patch(
                "scraper.urlopen",
                return_value=response,
            ),
        ):
            date_time.now.return_value = fixed_now
            result = scraper.search_xquik_mentions("Acme", limit=10, days_back=7)

        self.assertIsNotNone(result)
        self.assertEqual(result.iloc[0]["score"], 17)
        self.assertEqual(
            result.iloc[0]["date"],
            pd.Timestamp("2026-07-16T12:00:00"),
        )

    def test_redirect_handler_refuses_redirects(self):
        self.assertIsNone(
            scraper.RejectRedirects().redirect_request(
                object(),
                None,
                302,
                "Found",
                {},
                "https://attacker.invalid/capture",
            )
        )

    def test_default_opener_rejects_redirects_and_honors_timeout(self):
        request = object()
        response = MagicMock()
        with patch("scraper.build_opener") as build_opener:
            build_opener.return_value.open.return_value = response

            self.assertIs(scraper.urlopen(request, timeout=70), response)

        handler = build_opener.call_args.args[0]
        self.assertIsInstance(handler, scraper.RejectRedirects)
        build_opener.return_value.open.assert_called_once_with(request, timeout=70)

    def test_api_error_falls_back_to_next_source(self):
        with (
            patch.dict(
                "os.environ",
                {"XQUIK_API_KEY": "test-key"},
                clear=True,
            ),
            patch("scraper.urlopen", side_effect=OSError("offline")),
        ):
            result = scraper.search_xquik_mentions("Acme")

        self.assertIsNone(result)

    def test_skips_api_without_key(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("scraper.urlopen") as open_url,
        ):
            result = scraper.search_xquik_mentions("Acme")

        self.assertIsNone(result)
        open_url.assert_not_called()

    def test_uses_api_after_missing_csv_override(self):
        expected = pd.DataFrame([{"text": "Acme mention"}])
        with (
            patch(
                "scraper.load_xquik_csv_mentions",
                return_value=None,
            ) as load_csv,
            patch(
                "scraper.search_xquik_mentions",
                return_value=expected,
            ) as search_api,
        ):
            result = scraper.scrape_twitter("Acme", limit=20, days_back=14)

        self.assertIs(result, expected)
        load_csv.assert_called_once_with("Acme", limit=20, days_back=14)
        search_api.assert_called_once_with("Acme", limit=20, days_back=14)

    def test_csv_override_precedes_api(self):
        expected = pd.DataFrame([{"text": "Reviewed mention"}])
        with (
            patch(
                "scraper.load_xquik_csv_mentions",
                return_value=expected,
            ),
            patch("scraper.search_xquik_mentions") as search_api,
        ):
            result = scraper.scrape_twitter("Acme")

        self.assertIs(result, expected)
        search_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
