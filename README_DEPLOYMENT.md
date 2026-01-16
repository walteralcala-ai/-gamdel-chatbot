# 🤖 CHATBOT GAMDEL - DESPLIEGUE PERMANENTE

## ¿Qué es esto?

Un chatbot inteligente que responde preguntas sobre tus documentos PDF. Está listo para desplegar en producción.

## ¿Qué incluye?

✅ Chatbot con interfaz moderna  
✅ Búsqueda inteligente en documentos  
✅ Respuestas sin alucinaciones  
✅ Subida de múltiples PDFs  
✅ Historial de conversaciones  
✅ Meta-preguntas del sistema  
✅ Generación de tablas de documentos  

## Despliegue RÁPIDO (5 minutos)

### Opción A: Render (GRATIS)

1. Crea cuenta en https://render.com
2. Conecta tu GitHub
3. Crea nuevo "Web Service"
4. Selecciona este repositorio
5. Agrega variable: `OPENAI_API_KEY=tu_clave`
6. ¡Click Deploy!

**Resultado**: Tu chatbot estará online en `https://tu-app.onrender.com`

### Opción B: Docker local

```bash
# 1. Crear archivo .env
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Ejecutar con Docker
docker-compose up -d

# 3. Acceder a http://localhost:8000
```

### Opción C: Servidor propio

```bash
# 1. Clonar
git clone https://github.com/tu-usuario/gamdel-chatbot.git
cd gamdel-chatbot

# 2. Crear .env
cp .env.example .env
# Editar .env con tu clave de OpenAI

# 3. Ejecutar
docker-compose up -d

# 4. Acceder a http://tu-servidor:8000
```

## Archivos importantes

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Aplicación principal |
| `Dockerfile` | Para containerizar |
| `docker-compose.yml` | Para ejecutar con Docker |
| `DEPLOYMENT_GUIDE.md` | Guía detallada |
| `.env.example` | Variables de entorno |

## Configuración

### Variables de entorno

```bash
OPENAI_API_KEY=sk-...  # Requerido
PORT=8000              # Opcional
HOST=0.0.0.0          # Opcional
```

### Obtener clave de OpenAI

1. Ve a https://platform.openai.com/api-keys
2. Crea una nueva clave
3. Cópiala en tu `.env`

## Documentos

Los documentos se guardan en `/data`:
- Automáticamente se indexan
- Se pueden subir desde la interfaz
- Se eliminan desde la interfaz

## Monitoreo

```bash
# Ver logs
docker logs gamdel-chatbot -f

# Reiniciar
docker restart gamdel-chatbot

# Detener
docker stop gamdel-chatbot
```

## Características

### Chat inteligente
- Busca en tus documentos
- Responde preguntas específicas
- Cita la fuente correctamente
- No alucina

### Gestión de documentos
- Subir múltiples PDFs
- Ver lista de documentos
- Eliminar documentos individuales
- Eliminar todos de una vez

### Meta-preguntas
- "¿Cuántos documentos tengo?"
- "¿Cuántas páginas en total?"
- "Prepara un cuadro de documentos"

### Interfaz
- Chat tipo WhatsApp
- Mensajes ordenados cronológicamente
- Timestamps en cada mensaje
- Versión y fecha de documentos
- Enter para enviar

## Soporte

Si tienes problemas:
1. Revisa `DEPLOYMENT_GUIDE.md`
2. Verifica los logs
3. Asegúrate de tener clave de OpenAI válida

## Licencia

Uso interno GAMDEL E.I.R.L.

---

**¿Listo para desplegar? ¡Sigue la guía rápida arriba!** 🚀
