# ⚡ INICIO RÁPIDO - 3 PASOS

## PASO 1: Obtener clave de OpenAI (2 minutos)

1. Ve a https://platform.openai.com/api-keys
2. Haz login o crea cuenta
3. Click "Create new secret key"
4. Copia la clave (empieza con `sk-`)
5. **GUARDA ESTA CLAVE EN UN LUGAR SEGURO**

## PASO 2: Descargar y configurar (1 minuto)

1. Descarga `gamdel-chatbot-deployment.zip`
2. Descomprime en tu carpeta
3. Abre `gamdel-chatbot/` en terminal
4. Copia `cp .env.example .env`
5. Edita `.env` y pega tu clave:
   ```
   OPENAI_API_KEY=sk-tu-clave-aqui
   ```
6. Guarda el archivo

## PASO 3: Ejecutar (1 minuto)

### Opción A: Docker (Recomendado)
```bash
docker-compose up -d
```
Accede a: http://localhost:8000

### Opción B: Script automático
```bash
bash deploy.sh
```

### Opción C: Render (Online permanente)
1. Sube el código a GitHub
2. Ve a https://render.com
3. Conecta tu GitHub
4. Crea "Web Service"
5. Agrega variable: `OPENAI_API_KEY=sk-...`
6. ¡Click Deploy!

---

## ✅ ¡LISTO!

Tu chatbot está corriendo. Ahora:

1. **Sube documentos PDF** en la sección "Gestión"
2. **Haz preguntas** en el chat
3. **Disfruta** las respuestas inteligentes

---

## 🆘 Problemas?

### No funciona Docker
```bash
# Instala Docker desde: https://docs.docker.com/get-docker/
```

### Error de clave API
- Verifica que la clave sea correcta
- Asegúrate de no tener espacios
- Prueba en https://platform.openai.com/account/api-keys

### Puerto 8000 en uso
```bash
# Usa otro puerto
docker-compose.yml → cambiar "8000:8000" a "9000:8000"
```

---

## 📚 Documentación completa

- `README_DEPLOYMENT.md` - Guía detallada
- `DEPLOYMENT_GUIDE.md` - Todas las opciones
- `PRODUCTION.md` - Configuración avanzada

---

**¿Necesitas ayuda? Revisa los logs:**
```bash
docker logs gamdel-chatbot -f
```
