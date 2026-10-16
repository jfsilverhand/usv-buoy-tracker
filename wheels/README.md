# wheels/

Opsiyonel. Jetson'da önceden derlenmiş `pycuda-*-linux_aarch64.whl` buraya konursa
Dockerfile pip'ten derlemek yerine (~10 dk) onu kurar. Wheel'ler `.gitignore`'da;
bu klasörün repoda kalması için bu dosya var.

Wheel üretmek (Jetson host'ta):

```bash
pip3 wheel pycuda --no-deps -w wheels/
```
