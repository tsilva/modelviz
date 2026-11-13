"""Verify pinned trusted Production synchronization with fake values only."""

import json
import subprocess
import unittest
from unittest.mock import patch

from common import SecretError
from sync_production import Production

PROJECT = '362c6be8-ba92-4bdb-92ab-4305314517eb'
VALUE = "dummy-only '$HOME' \"quoted\"\\backslash\nsecond line\n"


def row(value=VALUE, **overrides):
    return {"key": "SENTRY_AUTH_TOKEN", "value": value, "workspace": PROJECT,
            "secretPath": "/", "type": "shared", **overrides}


class SecretTests(unittest.TestCase):
    def setUp(self):
        self.client = Production()


    def test_destination_identity_and_duplicates_are_rejected(self):
        for rows in ([row(workspace="other-project")], [row(secretPath="/other")],
                     [row(type="personal")], [row(), row()]):
            with patch.object(self.client, "command", return_value=json.dumps(rows)):
                with self.assertRaises(SecretError):
                    self.client.read()

    def test_provider_error_details_never_escape(self):
        response = subprocess.CompletedProcess([], 1, VALUE, VALUE)
        with patch("sync_production.subprocess.run", return_value=response):
            with self.assertRaises(SecretError) as error:
                self.client.read()
        self.assertNotIn(VALUE, str(error.exception))


    def test_manual_sync_refuses_a_combined_destination_before_writing(self):
        import sync_production as syncer
        with patch("sync_production.vercel", return_value={"envs":[{"key":"SENTRY_AUTH_TOKEN","target":["production","preview"]}]}) as api, \
                patch.object(self.client,"read",return_value={"SENTRY_AUTH_TOKEN":VALUE}):
            with self.assertRaisesRegex(SecretError,"no writes"):
                syncer.sync(self.client)
            self.assertEqual(api.call_count,1)

    def test_manual_sync_payload_is_stdin_only_and_error_body_is_suppressed(self):
        import sync_production as syncer
        response=subprocess.CompletedProcess([],1,VALUE,VALUE)
        with patch("sync_production.subprocess.run",return_value=response) as execute:
            with self.assertRaises(SecretError) as caught:
                syncer.vercel("POST",{"key":"SENTRY_AUTH_TOKEN","value":VALUE,"type":"sensitive","target":["production"]})
        args,kwargs=execute.call_args
        self.assertNotIn(VALUE," ".join(args[0]))
        self.assertEqual(json.loads(kwargs["input"])["value"],VALUE)
        self.assertIn("--input",args[0])
        self.assertNotIn(VALUE,str(caught.exception))

    def test_manual_sync_cannot_retarget_using_arguments(self):
        import sync_production as syncer
        with patch("sync_production.sync") as execute:
            with self.assertRaisesRegex(SecretError,"overrides"):
                syncer.main(["--env","dev"])
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
