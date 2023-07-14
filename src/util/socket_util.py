import queue
import socket
import pickle
import sys


import socket
import pickle
import threading


class Server:
    def __init__(self, host="localhost", port=1234,
                 n_clients=1, buffer_size=1024):
        super(Server, self).__init__()
        self.host = host
        self.port = port
        self.n_clients = n_clients
        self.buffer_size = buffer_size
        self.server_socket = None
        self.client_sockets = []

    def setup(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.n_clients)
        print(
            f"Server started on {self.host}:{self.port}. Waiting for {self.n_clients} client(s) to connect...")

        for _ in range(self.n_clients):
            client_socket, client_address = self.server_socket.accept()
            self.client_sockets.append(client_socket)
            print("Client connected:", client_address)

    def send(self, message):
        serialized_message = pickle.dumps(message)
        message_size = len(serialized_message)

        def send_message(client_socket):
            client_socket.sendall(pickle.dumps(message_size))

            total_sent = 0
            while total_sent < message_size:
                sent = client_socket.send(serialized_message[total_sent:])
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent

        threads = [threading.Thread(target=send_message, args=(
            client_socket,)) for client_socket in self.client_sockets]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def receive(self):
        def receive_data(client_socket, received_data_queue):
            data = client_socket.recv(self.buffer_size)
            data_size = pickle.loads(data)

            received_data_per_socket = b""
            while len(received_data_per_socket) < data_size:
                remaining_data = data_size - len(received_data_per_socket)
                chunk = client_socket.recv(
                    min(self.buffer_size, remaining_data))
                if chunk == b"":
                    raise RuntimeError("Socket connection broken")
                received_data_per_socket += chunk

            received_data_queue.put(pickle.loads(received_data_per_socket))

        received_data_queue = queue.Queue()

        threads = [threading.Thread(target=receive_data, args=(
            client_socket, received_data_queue)) for client_socket in self.client_sockets]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        received_data = []
        print(received_data)
        while not received_data_queue.empty():
            received_data.append(received_data_queue.get())

        return received_data if len(received_data) > 0 else None

    def close(self):
        for client_socket in self.client_sockets:
            client_socket.close()
        print("Server sockets closed.")


class Client:
    def __init__(self, host='localhost', port=1234, buffer_size=1024):
        super(Client, self).__init__()
        self.host = host
        self.port = port
        self.client_socket = None
        self.buffer_size = buffer_size

    def setup(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def send(self, message):
        serialized_message = pickle.dumps(message)
        message_size = len(serialized_message)
        self.client_socket.sendall(pickle.dumps(message_size))

        total_sent = 0
        while total_sent < message_size:
            sent = self.client_socket.send(serialized_message[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent


    def receive(self):
        data = self.client_socket.recv(self.buffer_size)
        data_size = pickle.loads(data)

        received_data = b""
        while len(received_data) < data_size:
            remaining_data = data_size - len(received_data)
            chunk = self.client_socket.recv(
                min(self.buffer_size, remaining_data))
            if chunk == b"":
                raise RuntimeError("Socket connection broken")
            received_data += chunk

        return pickle.loads(received_data)

    def close(self):
        self.client_socket.close()
        print("Client socket closed.")
