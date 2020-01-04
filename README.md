# TFM

En este repositorio se presenta el código utilizado para la elaboración del trabajo final de master:

- Título: Deep Learning para la detección de patologías de cáncer de piel y generación de imágenes de tejidos humanos
- Autor: Sandra Redondo Hernández
- Directores: Ferran Reverter Comes y Esteban Vegas Lozano
- Departamento: Genética, Microbiología y Estadística. Sección Estadística.
- Universidad: Universidad Politécnica de Catalunya
- Convocatoria: Enero 2020

Características técnicas:

-	Nvida Geforce RTX 2060 6GB
-	Intel Core i7-9759H 2.60GHz
-	RAM 16GB
-	HDD SSD 1TB
-	Windows 10 sistema 64bits
-	Python 3.6.8
-	Tensorﬂow 2-gpu

<b>Primer paso:</b> Crear el entorno en anaconda
<i>conda create -n new environment --file req.txt</i>

Si utilizas pip:

<i>env1/bin/pip freeze > root-spec.txt</i>

<i>env2/bin/pip install -r root-spec.txt</i>

<b>Segundo paso:</b> Descargar todas las imagenes del archivo ISIC https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery
