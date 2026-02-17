#!/bin/bash
# Script to create GitHub repository and push code
# Usage: ./create_github_repo.sh [repository-name]

REPO_NAME="${1:-prhp-framework}"
GITHUB_USER="sanjivakyosan"

echo "Creating GitHub repository: $GITHUB_USER/$REPO_NAME"
echo ""

# Check if GitHub token is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN environment variable not set."
    echo ""
    echo "To create the repo automatically, you need a GitHub Personal Access Token."
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Generate a new token with 'repo' scope"
    echo "3. Run: export GITHUB_TOKEN=your-token-here"
    echo "4. Then run this script again"
    echo ""
    echo "Alternatively, you can create the repo manually:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Description: PRHP Framework - Political Hierarchy Pruner with Qubit-Ready Ethical Emergence"
    echo "4. Choose Public or Private"
    echo "5. DO NOT initialize with README, .gitignore, or license (we already have these)"
    echo "6. Click 'Create repository'"
    echo "7. Then run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    exit 1
fi

# Create repository via GitHub API
echo "Creating repository via GitHub API..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"PRHP Framework - Political Hierarchy Pruner with Qubit-Ready Ethical Emergence\",\"private\":false}")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$REPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 201 ]; then
    echo "✅ Repository created successfully!"
    echo ""
    echo "Setting up git remote and pushing..."
    
    # Check if remote already exists
    if git remote get-url origin &>/dev/null; then
        echo "Remote 'origin' already exists. Updating..."
        git remote set-url origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    else
        git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    fi
    
    # Set main branch
    git branch -M main 2>/dev/null || git branch -M master
    
    # Push to GitHub
    echo "Pushing code to GitHub..."
    git push -u origin main 2>/dev/null || git push -u origin master
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo "Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
    else
        echo ""
        echo "⚠️  Push failed. You may need to authenticate."
        echo "Try: git push -u origin main"
    fi
elif [ "$HTTP_CODE" -eq 422 ]; then
    echo "⚠️  Repository '$REPO_NAME' may already exist."
    echo "Trying to add remote and push anyway..."
    
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git" 2>/dev/null || \
    git remote set-url origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    
    git branch -M main 2>/dev/null || git branch -M master
    git push -u origin main 2>/dev/null || git push -u origin master
else
    echo "❌ Failed to create repository. HTTP Code: $HTTP_CODE"
    echo "Response: $BODY"
    echo ""
    echo "Please create the repository manually at: https://github.com/new"
fi

