from flask import Flask, send_from_directory
import os


app = Flask(__name__)

@app.route("/runs/results/speech2face/<string:a>/samples/<string:d>.mp4")
def hello222(a, d):
    filepath = os.path.normpath('/' + os.path.join(a, 'samples')).lstrip('/')
    return send_from_directory(os.path.join('/runs/', filepath), d + '.mp4')

def start():
    import sys
    port = 5005
    if len(sys.argv) > 1:
        port = sys.argv[1]

    app.run('0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    start()
