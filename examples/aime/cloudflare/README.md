# AIME Cloudflare Sandbox bridge

This is a prepared-image deployment for `aime/aime@latest`. All 60 tasks use
the same Dockerfile-only Ubuntu image, so no task-specific image build or
upload is needed. Wrangler builds the image and pushes it to Cloudflare during
deployment. Containers use Standard-2, have no warm pool, and sleep after 30
minutes of inactivity.

Docker must be running locally. Wrangler must also be authenticated for the
target account; for non-interactive deployment, export `CLOUDFLARE_ACCOUNT_ID`
and `CLOUDFLARE_API_TOKEN`.

```bash
npm install
# This creates and immediately deploys a Worker version with the secret.
npx wrangler secret put SANDBOX_API_KEY
# This explicit deploy builds and pushes the prepared container image. Wrangler
# preserves the secret set above.
npm run deploy
```

Save the deployed Worker URL and bearer value as
`CLOUDFLARE_SANDBOX_API_URL` and `CLOUDFLARE_SANDBOX_API_KEY`. The Wrangler
deployment credential needs `Workers Scripts: Edit`, `Workers Containers:
Edit`, and `Account Settings: Read` for the target account.
