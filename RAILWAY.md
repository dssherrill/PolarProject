# Railway Deployment Guide

This guide explains how to deploy PolarProject to Railway.

## Prerequisites

- A Railway account (https://railway.app)
- This repository pushed to GitHub

## Deployment Files

The following files are configured for Railway deployment:

### 1. `requirements.txt`
Contains all Python dependencies with pinned versions. Railway automatically detects and installs these.

### 2. `Procfile`
Tells Railway how to start the application:
```
web: gunicorn polar_ui:server
```

### 3. `polar_ui.py`
Exposes the `server` variable required by gunicorn:
```python
server = app.server
```

### 4. `.python-version`
Specifies Python 3.12.3 for Railway's Python buildpack.

## Deployment Steps

### Option 1: Deploy from GitHub (Recommended)

1. Go to https://railway.app and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose the `dssherrill/PolarProject` repository
5. Railway will automatically:
   - Detect Python project
   - Install dependencies from `requirements.txt`
   - Use Python 3.12+ (specified in `pyproject.toml`)
   - Start the app using the `Procfile`

### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

## Environment Variables

No special environment variables are required for basic deployment. The app will automatically:
- Run in production mode when deployed
- Use port assigned by Railway (via `PORT` environment variable)
- Serve on `0.0.0.0` for external access

### Optional Environment Variables

- `PRODUCTION`: Set to `"true"` to force production mode
- `ENVIRONMENT`: Set to `"production"` to force production mode
- `PORT`: Automatically set by Railway (default: assigned by platform)

## Post-Deployment

After deployment:
1. Railway will provide a public URL
2. Access your app at: `https://your-app.up.railway.app`
3. The app will display glider performance charts

## Production Configuration

The app automatically configures for production when deployed:
- Debug mode disabled
- Runs on `0.0.0.0` (accepts external connections)
- Uses gunicorn WSGI server (included in requirements.txt)
- Optimized for production workloads

## Troubleshooting

### Build Fails
- Check that Python 3.12+ is available
- Verify all dependencies in `requirements.txt` are compatible

### App Won't Start
- Check Railway logs: `railway logs`
- Verify `Procfile` is in repository root
- Ensure `server` variable exists in `polar_ui.py`

### Port Issues
- Railway automatically sets `PORT` environment variable
- App detects production mode and binds to correct port

## Local Testing of Production Setup

Test the production configuration locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start with gunicorn (like Railway does)
gunicorn polar_ui:server

# Or test production mode directly
PRODUCTION=true python polar_ui.py
```

## Updates and Redeployment

Railway automatically redeploys when you push to the connected GitHub branch:

```bash
git add .
git commit -m "Update application"
git push
```

Railway will detect the push and redeploy automatically.

## Cost Considerations

- Railway offers a free tier with limitations
- Monitor usage in Railway dashboard
- Consider upgrading plan for production workloads

## Additional Resources

- Railway Documentation: https://docs.railway.app
- Railway Python Guide: https://docs.railway.app/guides/python
- Dash Deployment Guide: https://dash.plotly.com/deployment
