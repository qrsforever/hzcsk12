# K12AI


## 更新记录

构建镜像版本修改: `script/build_images.sh`

### K12AI

| 版本 | 说明 |
|:----:|:----:|
| 1.0.48 | first build |

### K12CV

| 版本 | 说明 |
|:----:|:----:|
| 1.0.48 | first build |

### K12NLP

| 版本 | 说明 |
|:----:|:----:|
| 1.0.48 | first build |


## 启动docker仓库服务

`/etc/docker/daemon.json`:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": ["https://registry.docker-cn.com"],
    "insecure-registries":["10.255.0.58:9500"]
}
```

```
sudo systemctl daemon-reload
sudo systemctl restart docker
docker pull registry
docker run -d -v /data/images:/var/lib/registry -p 9500:5000 --restart=always --name k12ai_images_registry registry
curl http://localhost:9500/v2/_catalog
```
