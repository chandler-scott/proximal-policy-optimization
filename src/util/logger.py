import os
import logging


class CustomLogger:
    _instance = None


    def __new__(cls, log_file=None, stats_file=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s\t|    %(message)s', '%H:%M:%S')

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cls._instance.logger.addHandler(console_handler)

            if log_file:
                cls._instance.create_output_folder()
                file_handler = logging.FileHandler(f'./out/{log_file}')
                file_handler.setFormatter(formatter)
                cls._instance.logger.addHandler(file_handler)

            if stats_file:
                
                cls._instance.create_output_folder()
                # Assign stats_file attribute
                cls._instance.stats_file = f'./out/{stats_file}'
                cls._instance.create_stats_file()  # Call create_stats_file method

        return cls._instance

    def debug(self, message) -> None:
        self.logger.debug(message)

    def info(self, message) -> None:
        self.logger.info(message)

    def warning(self, message) -> None:
        self.logger.warning(message)

    def error(self, message) -> None:
        self.logger.error(message)

    def critical(self, message) -> None:
        self.logger.critical(message)

    def create_output_folder(self) -> None:
        if not os.path.exists('./out'):
            os.makedirs('./out')

    def create_stats_file(self) -> None:
        with open(self.stats_file, 'w') as f:
            # Modify the header row as needed
            f.write('Average Reward\n')

    def statistics(self, message) -> None:
        try:
            numbers = [float(val) for val in message.split()
                       if val.replace('.', '').isdigit()]
            formatted_numbers = ', '.join(map(str, numbers))
            with open(self.stats_file, 'a') as f:
                f.write(formatted_numbers + '\n')
        except ValueError:
            self.logger.warning(
                "Invalid statistics message: {}".format(message))
