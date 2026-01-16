#!/bin/bash

echo "🚀 DEPLOYMENT SCRIPT - CHATBOT GAMDEL"
echo "======================================"

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado"
    echo "Instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker encontrado"

# Verificar .env
if [ ! -f .env ]; then
    echo "❌ Archivo .env no encontrado"
    echo "Copia .env.example a .env y configura tu OPENAI_API_KEY"
    exit 1
fi

echo "✅ Archivo .env encontrado"

# Construir imagen
echo "🔨 Construyendo imagen Docker..."
docker build -t gamdel-chatbot .

if [ $? -ne 0 ]; then
    echo "❌ Error construyendo imagen"
    exit 1
fi

echo "✅ Imagen construida"

# Detener contenedor anterior
echo "🛑 Deteniendo contenedor anterior..."
docker stop gamdel-chatbot 2>/dev/null
docker rm gamdel-chatbot 2>/dev/null

# Ejecutar nuevo contenedor
echo "▶️  Iniciando contenedor..."
docker run -d \
  -p 8000:8000 \
  --name gamdel-chatbot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  gamdel-chatbot

if [ $? -ne 0 ]; then
    echo "❌ Error iniciando contenedor"
    exit 1
fi

echo "✅ Contenedor iniciado"

# Esperar a que inicie
sleep 5

# Verificar que está corriendo
if docker ps | grep -q gamdel-chatbot; then
    echo ""
    echo "✅ ¡DEPLOYMENT EXITOSO!"
    echo ""
    echo "📍 Accede a: http://localhost:8000"
    echo ""
    echo "📋 Comandos útiles:"
    echo "   Ver logs:      docker logs gamdel-chatbot -f"
    echo "   Reiniciar:     docker restart gamdel-chatbot"
    echo "   Detener:       docker stop gamdel-chatbot"
else
    echo "❌ Error: El contenedor no está corriendo"
    docker logs gamdel-chatbot
    exit 1
fi
