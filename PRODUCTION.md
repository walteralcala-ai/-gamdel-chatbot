# 🏢 CONFIGURACIÓN DE PRODUCCIÓN

## Checklist antes de desplegar

- [ ] Clave de OpenAI válida y configurada
- [ ] Dockerfile probado localmente
- [ ] Variables de entorno configuradas
- [ ] Backups de documentos realizados
- [ ] Certificado SSL preparado
- [ ] Dominio configurado
- [ ] Firewall configurado

## Seguridad

### 1. Variables de entorno
```bash
# NUNCA commits esto en Git
.env
.env.local
.env.*.local
```

### 2. Firewall
```bash
# Permitir solo puertos necesarios
- 80 (HTTP)
- 443 (HTTPS)
- 8000 (Aplicación, solo si es necesario)
```

### 3. SSL/HTTPS
```bash
# Usar Let's Encrypt con Certbot
certbot certonly --standalone -d tu-dominio.com
```

### 4. Nginx reverse proxy
```nginx
server {
    listen 443 ssl;
    server_name tu-dominio.com;

    ssl_certificate /etc/letsencrypt/live/tu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tu-dominio.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoreo

### Logs
```bash
# Ver logs en tiempo real
docker logs -f gamdel-chatbot

# Guardar logs
docker logs gamdel-chatbot > chatbot.log 2>&1
```

### Health check
```bash
# Verificar que el chatbot responde
curl http://localhost:8000/documents?tenant=demo
```

## Backups

### Documentos
```bash
# Backup diario
0 2 * * * tar -czf /backups/gamdel-$(date +\%Y\%m\%d).tar.gz /app/data
```

## Mantenimiento

### Actualizar código
```bash
git pull
docker-compose down
docker-compose up -d --build
```

### Limpiar recursos
```bash
# Remover imágenes no usadas
docker image prune -a

# Remover volúmenes no usados
docker volume prune
```

## Troubleshooting

### Memoria
```bash
# Limitar memoria
docker run -m 512m gamdel-chatbot
```

### CPU
```bash
# Limitar CPU
docker run --cpus="1.5" gamdel-chatbot
```

## Soporte

Para problemas en producción:
1. Revisar logs: `docker logs gamdel-chatbot`
2. Verificar recursos: `docker stats`
3. Reiniciar: `docker restart gamdel-chatbot`
