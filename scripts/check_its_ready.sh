#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\Iniciando verificação de pré-build...\n"

ERROR_COUNT=0

echo -e "Verificando ferramentas..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[FALHA] Docker não encontrado.${NC}"
    ((ERROR_COUNT++))
else
    if ! docker info &> /dev/null; then
        echo -e "${RED}[FALHA] O daemon do Docker não está rodando.${NC}"
        ((ERROR_COUNT++))
    else
        echo -e "${GREEN}[OK] Docker está rodando.${NC}"
    fi
fi

FILES_TO_CHECK=(
    "docker-compose.yml"
    "docker/app/Dockerfile"
    "requirements.txt"
    ".env"
)

echo -e "\nVerificando arquivos..."
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}[OK] Arquivo encontrado: $file${NC}"
    else
        echo -e "${RED}[FALHA] Arquivo ausente: $file${NC}"
        ((ERROR_COUNT++))
    fi
done

echo -e "\nVerificando configurações (.env)..."
if [ -f ".env" ]; then
    if grep -q "GEMINI_API_KEY" .env; then
        echo -e "${GREEN}[OK] GEMINI_API_KEY encontrada no .env${NC}"
    else
        echo -e "${YELLOW}[AVISO] GEMINI_API_KEY não encontrada no .env. A aplicação pode falhar ao iniciar. Defina a chave API do Gemini.${NC}"
    fi
else
    echo -e "${RED}[FALHA] Arquivo .env não existe. O docker-compose irá falhar.${NC}"
fi

echo -e "\nVerificando diretórios de volume..."
VOLUMES=("src" "models" "data" "data/falkordb" "data/redis_cache")

for vol in "${VOLUMES[@]}"; do
    if [ -d "$vol" ]; then
        echo -e "${GREEN}[OK] Diretório existe: $vol${NC}"
    else
        echo -e "${YELLOW}[AVISO] Diretório '$vol' não existe. O Docker irá criá-lo como 'root', o que pode causar problemas de permissão.${NC}"
        echo -e "       Sugestão: mkdir -p $vol"
    fi
done

echo -e "\n----------------------------------------"
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}Tudo pronto! Você pode rodar: docker compose up --build${NC}"
    exit 0
else
    echo -e "${RED}Foram encontrados $ERROR_COUNT erro(s). Corrija-os antes de prosseguir.${NC}"
    exit 1
fi
