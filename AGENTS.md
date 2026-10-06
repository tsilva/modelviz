# ModelViz

Model parsing runs in the browser. Preserve the existing local-model endpoint and dependency boundary checks.

Default dev uses Infisical modelviz Development / through scripts/infisical/run.py. Production uses isolated modelviz-production Production /. The manual secrets:sync:production command copies only the build token to the pinned Vercel production project and requires redeployment. Never print credentials or upload private dotenv files; retain Keychain originals until replacement is verified.
