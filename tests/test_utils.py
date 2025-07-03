import pytest
from unittest.mock import patch, Mock
import os
import signal
import psutil

from core.utils import terminate_existing_processes

@patch('psutil.process_iter')
@patch('os.getpid', return_value=12345)
def test_terminate_existing_processes(mock_getpid, mock_process_iter):
    # Mock a process that should be terminated
    mock_other_process = Mock()
    mock_other_process.info = {'pid': 54321, 'name': 'python', 'cmdline': ['python', 'main.py']}
    mock_other_process.is_running.return_value = False # Simulate graceful termination

    # Mock the current process
    mock_current_process = Mock()
    mock_current_process.info = {'pid': 12345, 'name': 'python', 'cmdline': ['python', 'main.py']}

    # Mock another unrelated process
    mock_unrelated_process = Mock()
    mock_unrelated_process.info = {'pid': 98765, 'name': 'python', 'cmdline': ['python', 'other_script.py']}

    mock_process_iter.return_value = [mock_other_process, mock_current_process, mock_unrelated_process]

    terminate_existing_processes()

    # Assert that terminate was called for the other main.py process
    mock_other_process.terminate.assert_called_once()
    mock_other_process.wait.assert_called_once_with(timeout=5)
    mock_other_process.kill.assert_not_called() # Should not be called if terminate is graceful

    # Assert that terminate was NOT called for the current process or other_script.py
    mock_current_process.terminate.assert_not_called()
    mock_unrelated_process.terminate.assert_not_called()

@patch('psutil.process_iter')
@patch('os.getpid', return_value=12345)
def test_terminate_existing_processes_no_other_instances(mock_getpid, mock_process_iter):
    # Simulate no other instances running
    mock_current_process = Mock()
    mock_current_process.info = {'pid': 12345, 'name': 'python', 'cmdline': ['python', 'main.py']}
    mock_process_iter.return_value = [mock_current_process]

    terminate_existing_processes()

    # Ensure terminate was not called
    mock_current_process.terminate.assert_not_called()

@patch('psutil.process_iter')
@patch('os.getpid', return_value=12345)
def test_terminate_existing_processes_psutil_failure(mock_getpid, mock_process_iter):
    # Test that it handles psutil failure gracefully
    mock_process_iter.side_effect = Exception("psutil error")
    with pytest.raises(Exception) as excinfo:
        terminate_existing_processes()
    assert "psutil error" in str(excinfo.value)
    # No exception should be raised, and psutil.process_iter should be called once
    mock_process_iter.assert_called_once()

@patch('psutil.process_iter')
@patch('os.getpid', return_value=12345)
def test_terminate_existing_processes_kill_on_timeout(mock_getpid, mock_process_iter):
    # Mock a process that does not terminate gracefully
    mock_other_process = Mock()
    mock_other_process.info = {'pid': 54321, 'name': 'python', 'cmdline': ['python', 'main.py']}
    mock_other_process.is_running.return_value = True # Simulate process still running after terminate

    mock_process_iter.return_value = [mock_other_process]

    terminate_existing_processes()

    # Assert that terminate and kill were called
    mock_other_process.terminate.assert_called_once()
    mock_other_process.wait.assert_called_once_with(timeout=5)
    mock_other_process.kill.assert_called_once()