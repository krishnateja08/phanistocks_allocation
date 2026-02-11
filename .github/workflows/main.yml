name: Portfolio Analysis Report

on:
  schedule:
    # Runs every Sunday at 7:00 PM CST (which is Monday 1:00 AM UTC)
    # CST is UTC-6, so 7:00 PM CST = 1:00 AM UTC next day
    - cron: '0 1 * * 1'  # Every Monday at 01:00 UTC = Sunday 7:00 PM CST
  
  workflow_dispatch:  # Allows manual trigger from GitHub UI

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  generate-report:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install yfinance pandas numpy pytz
    
    - name: Generate portfolio report
      env:
        SENDER_EMAIL: ${{ secrets.SENDER_EMAIL }}
        SENDER_PASSWORD: ${{ secrets.SENDER_PASSWORD }}
        RECIPIENT_EMAIL: ${{ secrets.RECIPIENT_EMAIL }}
      run: |
        python portfolio_analyzer.py
    
    - name: Commit and push changes
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add index.html portfolio_analysis_*.html portfolio_data_*.json
        git diff --quiet && git diff --staged --quiet || (git commit -m "Update portfolio report - $(date +'%Y-%m-%d %H:%M:%S')" && git push)
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: .
        publish_branch: gh-pages
        keep_files: false
        user_name: 'github-actions[bot]'
        user_email: 'github-actions[bot]@users.noreply.github.com'
