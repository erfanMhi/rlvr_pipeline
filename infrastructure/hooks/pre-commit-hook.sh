#!/bin/bash

echo "Running pre-commit hook..."

# Patterns for sensitive files
SENSITIVE_PATTERNS=(
  "terraform.tfvars$"
  "terraform.tfstate"
  "terraform.tfstate.backup"
  ".terraform/"
  "*.pem"
  "id_rsa"
  "id_ed25519"
  "*.pkcs8"
  "*.ppk"
  "*.key"
  "*secret*"
  "*password*"
  ".env"
)

# Check for sensitive files
FOUND_SENSITIVE=0
for pattern in "${SENSITIVE_PATTERNS[@]}"; do
  # Find files matching pattern that are staged
  FILES=$(git diff --cached --name-only | grep -E "$pattern" || true)
  if [ -n "$FILES" ]; then
    echo "ERROR: Attempting to commit sensitive file matching pattern: $pattern"
    echo "Files: $FILES"
    FOUND_SENSITIVE=1
  fi
done

# Check for potential hardcoded credentials in code
CREDENTIAL_PATTERNS=(
  "access_key"
  "secret_key"
  "password"
  "token"
  "apikey"
  "api_key"
  "tenant_id.*="
  "project_id.*="
)

for pattern in "${CREDENTIAL_PATTERNS[@]}"; do
  # Find lines in staged files that might contain credentials
  MATCHES=$(git diff --cached -G "$pattern" | grep -v "\.example" | grep -E "^\+" | grep -E "$pattern" || true)
  if [ -n "$MATCHES" ]; then
    echo "WARNING: Potential credentials found in staged changes:"
    echo "$MATCHES"
    echo ""
    echo "Please verify these are not actual credentials before committing."
    # Uncomment to make this a hard error:
    # FOUND_SENSITIVE=1
  fi
done

if [ $FOUND_SENSITIVE -eq 1 ]; then
  echo "Pre-commit hook failed due to sensitive files or data."
  echo "If you're ABSOLUTELY SURE this is intentional, you can bypass with: git commit --no-verify"
  exit 1
fi

echo "Pre-commit hook passed."
exit 0

# Fix trailing whitespace and end-of-file issues
# find infrastructure/nebius -name "*.tf" -o -name "*.tftpl" -o -name "*.example" | xargs sed -i '' -e 's/[[:space:]]*$//'

# # Add newlines at end of files if missing
# find infrastructure/nebius -name "*.tf" -o -name "*.tftpl" -o -name "*.example" | while read file; do
#   [ "$(tail -c 1 "$file")" != "" ] && echo "" >> "$file"
# done
