import queue
import socket
import pickle
import sys
import logging
import time
import socket
import pickle
import threading

from util.logger import CustomLogger


def send_receive_with_timeout(send_func, receive_func, timeout, max_retries):
    retries = 0

    while retries < max_retries:
        try:
            send_func()  # Send the message
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = receive_func()  # Attempt to receive a response

                if response is not None:
                    return response  # Return the response if received successfully

            retries += 1  # Increment the retry count
        except Exception as e:
            raise Exception
            logging.exception('send_rcv error!')

    raise TimeoutError(f"No response received after {max_retries} retries")


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
        error_flag = threading.Event()  # Create a threading.Event object

        def send_message(client_socket):
            try:
                client_socket.sendall(pickle.dumps(message_size))
                client_socket.recv(self.buffer_size)

                # send two copies for peace of mind!
                client_socket.sendall(serialized_message)
                client_socket.sendall(serialized_message)


            except Exception as e:
                error_flag.set()  # Set the error flag if an exception occurs
                print(f'Error occurred: {e}')

        threads = [threading.Thread(target=send_message, args=(
            client_socket,)) for client_socket in self.client_sockets]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        if error_flag.is_set():  # Check if any thread encountered an error
            print("Exiting due to an error.")
            raise Exception


    def receive(self):
        def receive_data(client_socket, received_data_queue, error_flag):
            try:
                data = client_socket.recv(self.buffer_size)
                data_size = pickle.loads(data)
                client_socket.send(pickle.dumps('ACK'))
                data = client_socket.recv(data_size)

                received_data_queue.put(pickle.loads(data))
            except Exception as e:
                error_flag.set()  # Set the error flag if an exception occurs
                print(f'Error occurred: {e}')

        received_data_queue = queue.Queue()
        error_flag = threading.Event()  # Create a threading.Event object

        threads = [threading.Thread(target=receive_data, args=(
            client_socket, received_data_queue, error_flag)) for client_socket in self.client_sockets]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        if error_flag.is_set():  # Check if any thread encountered an error
            print("Exiting due to an error.")
            # Add any necessary cleanup or exit logic here

        received_data = []
        while not received_data_queue.empty():
            received_data.append(received_data_queue.get())

        return received_data if len(received_data) > 0 else None


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
        try:
            serialized_message = pickle.dumps(message)
            message_size = len(serialized_message)
            self.client_socket.sendall(pickle.dumps(message_size))
            self.client_socket.recv(self.buffer_size)
            self.client_socket.sendall(serialized_message)

        except:
            raise Exception

    def receive(self):
        data = self.client_socket.recv(self.buffer_size)
        data_size = pickle.loads(data)
        self.client_socket.send(pickle.dumps('ACK'))

        data = self.client_socket.recv(data_size)
        data_retry = self.client_socket.recv(data_size)

        try:
            models = pickle.loads(data)
        except:
            CustomLogger().warning(f'First model recieved was bogus..\nTrying to load second one.')
            models = pickle.loads(data_retry)

        return models

    def close(self):
        self.client_socket.close()
        print("Client socket closed.")
