# ForceTrack Bar Path API

FastAPI + OpenCV server for server-side barbell tracking.

## Deploy to Railway

### Option A: GitHub (recommended)
1. Create a new GitHub repo and push these 3 files (main.py, requirements.txt, Dockerfile)
2. In Railway: New Project → Deploy from GitHub repo → select your repo
3. Railway auto-detects the Dockerfile and builds it

### Option B: Railway CLI
```bash
npm install -g @railway/cli
cd barpath-api
railway login
railway init
railway up
```

## Environment Variables (set in Railway → Variables tab)
- `BARPATH_API_KEY` = any secret string you choose (e.g. `forgedfit-2024`)

## Get Your Domain
Railway → your service → Settings → Networking → Generate Domain
Copy the URL (looks like `https://xxx.up.railway.app`)

## Connect to Netlify
In Netlify → forgedfitnesspt → Site configuration → Environment variables:
- `VITE_BARPATH_API_URL` = your Railway URL (e.g. `https://xxx.up.railway.app`)
- `VITE_BARPATH_API_KEY` = same secret string from above

Then trigger a Netlify redeploy.

## Test It
Visit: `https://your-railway-url.up.railway.app/health`
Should return: `{"status":"ok","version":"1.0"}`
