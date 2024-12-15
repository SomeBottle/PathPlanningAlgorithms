FROM somebottle/astarbase:0.0.1

# Binder 构建时用的镜像

USER somebottle

COPY --chown=somebottle:somebottle . /home/somebottle