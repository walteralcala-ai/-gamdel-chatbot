#!/bin/bash

echo "🔍 VERIFICACIÓN DE INSTALACIÓN"
echo "=============================="
echo ""

# Verificar Docker
echo "1️⃣  Verificando Docker..."
if command -v docker &> /dev/null; then
    echo "   ✅ Docker instalado: $(docker --version)"
else
    echo "   ❌ Docker NO instalado"
    echo "   Instala desde: https://docs.docker.com/get-docker/"
fi

echo ""

# Verificar .env
echo "2️⃣  Verificando .env..."
if [ -f .env ]; then
    echo "   ✅ Archivo .env encontrado"
    if grep -q "OPENAI_API_KEY" .env; then
        echo "   ✅ OPENAI_API_KEY configurada"
    else
        echo "   ❌ OPENAI_API_KEY NO configurada"
    fi
else
    echo "   ❌ Archivo .env NO encontrado"
    echo "   Copia: cp .env.example .env"
fi

echo ""

# Verificar archivos necesarios
echo "3️⃣  Verificando archivos..."
files=("app.py" "Dockerfile" "docker-compose.yml" "requirements.txt")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file NO encontrado"
    fi
done

echo ""

# Verificar contenedor
echo "4️⃣  Verificando contenedor..."
if docker ps | grep -q gamdel-chatbot; then
    echo "   ✅ Contenedor corriendo"
    echo "   📍 Accede a: http://localhost:8000"
else
    echo "   ⏸️  Contenedor no está corriendo"
    echo "   Inicia con: docker-compose up -d"
fi

echo ""
echo "✅ Verificación completada"
