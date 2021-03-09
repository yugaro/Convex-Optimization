FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	sudo\
	wget\
	vim\
	python3\
	python3-pip\
	tzdata\
	libx11-dev\
	python3-tk
ENV DISPLAY="host.docker.internal:0.0"

RUN pip3 install numpy matplotlib cvxpy cvxopt MOSEK
ENV MOSEKLM_LICENSE_FILE="/usr/local/lib/python3.8/dist-packages/mosek"
COPY mosek.lic /usr/local/lib/python3.8/dist-packages/mosek

CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]

# This docker image is created for python users to optimize convex problems and has the minimum needed packages(numpy, matplotlib, cvxpy, cvxopt, mosek).
# The Default command of this image is ["jupyter","lab"] (for jupyter users) .
# ex) docker run --rm -it -p 8888:8888 -v ~/host/path/to/file/or/directory:/new_dir yugaro/opt-python
# If you want to plot your data with your host GUI, you shuold use command ["bash"] and open XQuratz (for Mac users).
# ex) docker run --rm -it -v ~/host/path/to/file/or/directory:/new_dir yugaro/opt-python bash
