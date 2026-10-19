"""Byte bridge between a TCP port on 127.0.0.1 and a Unix socket.

    tcp2unix PORT PATH   listen on TCP PORT, connect each client to the socket PATH
    unix2tcp PATH PORT   listen on the socket PATH, connect each client to TCP PORT

How a program OUTSIDE a private network namespace (`unshare -rn`) reaches a
node inside it: the namespace shares the filesystem, not the network, so one
bridge inside listens on a socket and one outside listens on a port
(`examples/mcp_client_rig.sh`). A socket path is limited to 108 bytes — keep it
short (`/tmp/…`), never under a deep scratch directory.
"""
import asyncio
import sys


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
        # Pass the half-close on, so a reply still flows the other way.
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        pass


async def main(mode, listen_on, connect_to):
    async def handle(reader, writer):
        try:
            if mode == "tcp2unix":
                up_r, up_w = await asyncio.open_unix_connection(connect_to)
            else:
                up_r, up_w = await asyncio.open_connection("127.0.0.1", int(connect_to))
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))
        for w in (writer, up_w):
            w.close()

    if mode == "tcp2unix":
        server = await asyncio.start_server(handle, "127.0.0.1", int(listen_on))
    elif mode == "unix2tcp":
        server = await asyncio.start_unix_server(handle, listen_on)
    else:
        sys.exit(f"mode must be tcp2unix or unix2tcp, not {mode}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    asyncio.run(main(*sys.argv[1:4]))
