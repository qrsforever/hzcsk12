docker run -itd --name privoxy --cap-add=NET_ADMIN --volume ./privoxy/user.action:/etc/privoxy/user.action --volume  ./privoxy/user.filter:/etc/privoxy/user.filter --publish 8118:8118 --restart always vimagick/privoxy
