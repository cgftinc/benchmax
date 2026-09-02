# AIME Cloudflare Sandbox bridge

This is a prepared-image deployment for `aime/aime@latest`. All 60 tasks use
the same Dockerfile-only Ubuntu image, so no task-specific image build or
upload is needed. Wrangler builds the image and pushes it to Cloudflare during
deployment. Containers use Standard-2, have no warm pool, and sleep after 30
minutes of inactivity.

```bash
npm install
npx wrangler secret put SANDBOX_API_KEY
npm run deploy
```

Save the deployed Worker URL and bearer value as
`CLOUDFLARE_SANDBOX_API_URL` and `CLOUDFLARE_SANDBOX_API_KEY`. The Wrangler
deployment credential needs `Workers Scripts: Edit`, `Workers Containers:
Edit`, and `Account Settings: Read` for the target account.
