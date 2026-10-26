#!/bin/bash
# Boot script: install Docker, render .env from SSM (KMS-encrypted), start the bot.
set -euo pipefail
dnf install -y docker git jq
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-aarch64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone --depth 1 --branch "${repo_ref}" "${repo_url}" /opt/kronos
cd /opt/kronos/deploy

cat > /usr/local/bin/kronos-render-env <<'EOS'
#!/bin/bash
# Pull /kronos/* from SSM into deploy/.env (root-only). Re-run after rotating a secret.
set -euo pipefail
umask 077
aws ssm get-parameters-by-path --region "$1" --path /kronos/ --with-decryption \
  --query 'Parameters[].[Name,Value]' --output text \
  | awk -F'\t' '{sub("/kronos/","",$1); if ($2 != "CHANGE_ME") print $1"="$2}' > /opt/kronos/deploy/.env.secrets
cat /opt/kronos/deploy/.env.base /opt/kronos/deploy/.env.secrets > /opt/kronos/deploy/.env
EOS
chmod 700 /usr/local/bin/kronos-render-env

cat > .env.base <<EOF
KRONOS_MODE=paper
KRONOS_HARD_CAP=25
KRONOS_IMAGE=${image}
EOF
[ -f config/config.yaml ] || cp config/config.example.yaml config/config.yaml
/usr/local/bin/kronos-render-env "${region}"

if grep -q '^GHCR_TOKEN=' .env; then
  grep '^GHCR_TOKEN=' .env | cut -d= -f2- | docker login ghcr.io -u kronos --password-stdin
fi
docker compose pull bot
docker compose up -d bot
