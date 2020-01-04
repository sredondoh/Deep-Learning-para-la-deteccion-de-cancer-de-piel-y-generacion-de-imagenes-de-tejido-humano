# TFM

En este repositorio se presenta el código utilizado para la elaboración del trabajo final de master:

- Título: Deep Learning para la detección de patologías de cáncer de piel y generación de imágenes de tejidos humanos
- Autor: Sandra Redondo Hernández
- Directores: Ferran Reverter Comes y Esteban Vegas Lozano
- Departamento: Genética, Microbiología y Estadística. Sección Estadística.
- Universidad: Universidad Politécnica de Catalunya
- Convocatoria: Enero 2020

Características técnicas:

	Nvida Geforce RTX 2060 6GB
	Intel Core i7-9759H 2.60GHz
	RAM 16GB
	HDD SSD 1TB
	Windows 10 sistema 64bits
	Python 3.6.8
	Tensorﬂow 2-gpu

<s>Primer paso:</s> Crear el entorno en anaconda
$ conda create -n new environment --file req.txt
Si utilizas pip:
$ env1/bin/pip freeze > root-spec.txt
$ env2/bin/pip install -r root-spec.txt

<s>Segundo paso:</s> Descargar todas las imagenes del archivo ISIC https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery
