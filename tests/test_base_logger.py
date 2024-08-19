from cv_expt.base.logger.base_logger import BaseLoggerConfig, BaseLogger


def test_local_logger():
    config = BaseLoggerConfig()
    logger = BaseLogger(config)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.log_metric("accuracy", 0.95, 1)
    assert True

    assert (config.log_path / config.log_file).exists()
    assert (config.log_path / config.metric_file).exists()
