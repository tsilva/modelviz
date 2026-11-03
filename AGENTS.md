# ModelViz

Model parsing runs in the browser. Preserve the existing local-model endpoint and dependency boundary checks.

Local development and builds skip Sentry source-map uploads. The live Vercel build handles uploads. Production uses isolated modelviz-production Production /. The manual secrets:sync:production command copies only the build token to the pinned Vercel production project and requires redeployment. Never print credentials or upload private dotenv files; retain Keychain originals until replacement is verified.
