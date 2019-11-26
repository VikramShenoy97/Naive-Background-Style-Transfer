from naive_background_style_transfer import NaiveBackgroundStyleTransfer
content = "Input_Images/portrait.jpg"
style = "Input_Images/scream.jpg"
nbst = NaiveBackgroundStyleTransfer(number_of_epochs=100, verbose = True)
nbst.perform(content, style)
nbst.generate_gif(speed=1.5)
