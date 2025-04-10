#!/bin/bash
# Start the application
gunicorn -w 1 web3flas:app --bind 0.0.0.0:$PORT