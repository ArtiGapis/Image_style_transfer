from flask import Flask, request, render_template
import transfer
import base64
import io


app = Flask(__name__)

# Create a route to handle the input form submission and convert the image to HTML:

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        # epoch = request.files['epoch']
        epoch = request.values['epoch']
        # Open the image using Pillow
        print(f'epochos: {epoch}')
        img = transfer.run_style_transfer(image, "images/style2.jpeg", epochs=int(epoch))
        print(f'image request type{type(image)}')
        # img = Image.open(img_convert)
        # Convert the image to HTML
        img_data = io.BytesIO()
        img.save(img_data, format='PNG')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode()
        # Render the HTML template with the converted image
        return render_template('transfer_page.html', img_base64=img_base64)
    else:
        return render_template('transfer_page.html')


if __name__ == '__main__':
    app.run()