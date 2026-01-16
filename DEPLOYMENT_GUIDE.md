# 🚀 GUÍA DE DEPLOYMENT - CHATBOT GAMDEL

## Opción 1: Desplegar GRATIS en Render (Recomendado)

### Paso 1: Preparar el repositorio
```bash
cd /home/ubuntu/gamdel-chatbot
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/TU_USUARIO/gamdel-chatbot.git
git push -u origin main
```

### Paso 2: Crear cuenta en Render
1. Ve a https://render.com
2. Regístrate con GitHub
3. Conecta tu repositorio

### Paso 3: Crear nuevo servicio
1. Click en "New +" → "Web Service"
2. Selecciona tu repositorio `gamdel-chatbot`
3. Configura:
   - **Name**: `gamdel-chatbot`
   - **Environment**: `Docker`
   - **Plan**: `Free`
   - **Region**: Elige la más cercana

### Paso 4: Agregar variables de entorno
1. En la sección "Environment"
2. Agrega: `OPENAI_API_KEY` = tu clave de OpenAI
3. Click en "Deploy"

**¡Listo! Tu chatbot estará online en ~5 minutos**

---

## Opción 2: Desplegar en tu propio servidor

### Requisitos
- Docker instalado
- Git instalado
- Acceso SSH al servidor

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/TU_USUARIO/gamdel-chatbot.git
cd gamdel-chatbot
```

2. **Crear archivo .env**
```bash
echo "OPENAI_API_KEY=tu_clave_aqui" > .env
```

3. **Construir la imagen Docker**
```bash
docker build -t gamdel-chatbot .
```

4. **Ejecutar el contenedor**
```bash
docker run -d \
  -p 8000:8000 \
  --name gamdel-chatbot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  gamdel-chatbot
```

5. **Verificar que está corriendo**
```bash
docker ps
curl http://localhost:8000
```

---

## Opción 3: Usar Docker Compose

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/TU_USUARIO/gamdel-chatbot.git
cd gamdel-chatbot
```

2. **Crear archivo .env**
```bash
echo "OPENAI_API_KEY=tu_clave_aqui" > .env
```

3. **Ejecutar con Docker Compose**
```bash
docker-compose up -d
```

4. **Ver logs**
```bash
docker-compose logs -f
```

5. **Detener**
```bash
docker-compose down
```

---

## Configurar dominio personalizado

### En Render
1. Ve a "Settings" → "Custom Domain"
2. Agrega tu dominio (ej: chatbot.tudominio.com)
3. Sigue las instrucciones de DNS

### En tu servidor
1. Configura Nginx o Apache como reverse proxy
2. Apunta el dominio a tu servidor
3. Configura SSL con Let's Encrypt

---

## Monitoreo y mantenimiento

### Ver logs
```bash
docker logs gamdel-chatbot -f
```

### Reiniciar
```bash
docker restart gamdel-chatbot
```

### Actualizar código
```bash
git pull
docker-compose down
docker build -t gamdel-chatbot .
docker-compose up -d
```

---

## Solución de problemas

### El chatbot no inicia
```bash
docker logs gamdel-chatbot
```

### Puerto 8000 en uso
```bash
docker run -d -p 9000:8000 gamdel-chatbot
# Acceder a http://localhost:9000
```

### Problemas con OPENAI_API_KEY
- Verifica que la clave sea correcta
- Asegúrate de que tiene permisos para gpt-4.1-mini
- Revisa los logs: `docker logs gamdel-chatbot`

---

## Variables de entorno disponibles

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Clave de API de OpenAI | `sk-...` |
| `PORT` | Puerto (default 8000) | `8000` |
| `HOST` | Host (default 0.0.0.0) | `0.0.0.0` |

---

## Seguridad

- ✅ Nunca commits la clave API en Git
- ✅ Usa variables de entorno
- ✅ Configura HTTPS en producción
- ✅ Usa firewall para limitar acceso
- ✅ Realiza backups regulares de `/data`

---

## Soporte

Si tienes problemas:
1. Revisa los logs: `docker logs gamdel-chatbot`
2. Verifica la conexión a OpenAI
3. Asegúrate de que los documentos se cargaron correctamente
