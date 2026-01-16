# 🤖 CHATBOT GAMDEL - VERSIÓN PRODUCCIÓN

Chatbot inteligente para responder preguntas sobre tus documentos PDF. **Listo para desplegar permanentemente.**

## 🎯 ¿Qué es?

Un asistente IA que:
- ✅ Lee tus documentos PDF
- ✅ Responde preguntas específicas
- ✅ Cita las fuentes correctamente
- ✅ NO alucina (no inventa respuestas)
- ✅ Interfaz moderna tipo WhatsApp
- ✅ Genera reportes automáticos

## ⚡ Inicio en 3 pasos

**Lee:** `INICIO_RAPIDO.md`

```bash
# 1. Configurar clave de OpenAI
cp .env.example .env
# Edita .env con tu clave

# 2. Ejecutar
docker-compose up -d

# 3. Acceder
# http://localhost:8000
```

## 📦 Contenido

```
gamdel-chatbot/
├── app.py                      # Aplicación principal
├── Dockerfile                  # Para containerizar
├── docker-compose.yml          # Ejecutar con Docker
├── requirements.txt            # Dependencias Python
├── .env.example               # Plantilla de variables
├── deploy.sh                  # Script automático
├── verify.sh                  # Verificar instalación
├── INICIO_RAPIDO.md           # ⭐ COMIENZA AQUÍ
├── README_DEPLOYMENT.md       # Guía rápida
├── DEPLOYMENT_GUIDE.md        # Guía detallada
└── PRODUCTION.md              # Configuración avanzada
```

## 🚀 Opciones de despliegue

### 1. Local (Docker)
```bash
docker-compose up -d
# http://localhost:8000
```

### 2. Render (GRATIS, online permanente)
1. Sube a GitHub
2. Ve a https://render.com
3. Conecta repositorio
4. Crea Web Service
5. Agrega `OPENAI_API_KEY`
6. Deploy

### 3. Tu servidor
```bash
bash deploy.sh
```

## 📋 Requisitos

- Docker (https://docs.docker.com/get-docker/)
- Clave de OpenAI (https://platform.openai.com/api-keys)
- 2GB RAM mínimo
- Conexión a internet

## 🎨 Características

### Chat inteligente
- Búsqueda en documentos
- Respuestas precisas
- Cita de fuentes
- Timestamps en mensajes

### Gestión de documentos
- Subir múltiples PDFs
- Ver lista de documentos
- Eliminar documentos
- Información de versión y fecha

### Meta-preguntas
- "¿Cuántos documentos tengo?"
- "¿Cuántas páginas en total?"
- "Prepara un cuadro de documentos"

### Interfaz
- Diseño moderno
- Responsive (funciona en móvil)
- Colores corporativos
- Fácil de usar

## 🔧 Configuración

### Variables de entorno
```bash
OPENAI_API_KEY=sk-...    # Requerido
PORT=8000                # Opcional
HOST=0.0.0.0            # Opcional
```

### Modelos soportados
- gpt-4.1-mini (recomendado)
- gpt-4.1-nano (más rápido)
- gemini-2.5-flash (alternativa)

## 📚 Documentación

| Documento | Descripción |
|-----------|-------------|
| **INICIO_RAPIDO.md** | Comienza aquí (3 pasos) |
| **README_DEPLOYMENT.md** | Guía rápida de despliegue |
| **DEPLOYMENT_GUIDE.md** | Todas las opciones |
| **PRODUCTION.md** | Configuración avanzada |

## 🆘 Troubleshooting

### Docker no instalado
```bash
# Instala desde: https://docs.docker.com/get-docker/
```

### Error de clave API
```bash
# Verifica en: https://platform.openai.com/api-keys
# Asegúrate de no tener espacios en .env
```

### Ver logs
```bash
docker logs gamdel-chatbot -f
```

### Reiniciar
```bash
docker restart gamdel-chatbot
```

## 📊 Estadísticas

- **Documentos**: 16 cargados
- **Caracteres**: 421,550
- **Páginas**: 205
- **Modelo**: gpt-4.1-mini
- **Temperatura**: 0.1 (respuestas consistentes)

## 🔒 Seguridad

- ✅ Clave API en variables de entorno
- ✅ NO se guarda en Git
- ✅ Containerizado con Docker
- ✅ Respuestas validadas
- ✅ Sin alucinaciones

## 🎓 Casos de uso

- Soporte técnico automatizado
- Asistente de documentación
- Análisis de procedimientos
- Respuestas a preguntas frecuentes
- Búsqueda inteligente de información

## 📞 Soporte

1. Revisa `INICIO_RAPIDO.md`
2. Consulta `DEPLOYMENT_GUIDE.md`
3. Verifica logs: `docker logs gamdel-chatbot`

## 📄 Licencia

Uso interno GAMDEL E.I.R.L.

---

**¿Listo? Comienza con `INICIO_RAPIDO.md` 🚀**
