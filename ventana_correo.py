# ventana_correo.py
from tkinter import *
from tkinter import filedialog, messagebox
from email.message import EmailMessage
from PIL import ImageTk, Image
import smtplib
import mimetypes
import os

class AplicacionCorreo:
    def __init__(self, root):
        self.ventana = root
        self.ventana.title("Enviar archivos")
        self.ventana.geometry("350x470")
        self.ventana.resizable(0, 0)
        self.ventana.config(bd=10,bg="#030075")

        self.archivo_adjunto = None
        self.crear_interfaz()

    def crear_interfaz(self):
        Label(self.ventana, text="ENVIAR CORREO VIA GMAIL", fg="white", bg="#030075", font=("Arial", 15, "bold"), padx=5, pady=5).grid(row=0, column=0, columnspan=2)

        imagen_gmail = Image.open("Assets/PNG/Asset 1.png")
        nueva_imagen = imagen_gmail.resize((125, 84))
        render = ImageTk.PhotoImage(nueva_imagen)
        Label(self.ventana, image=render).grid(row=1, column=0, columnspan=2)
        Label(self.ventana, image=render).image = render  # Guardar referencia

        self.destinatario = StringVar()
        self.asunto = StringVar()

        Label(self.ventana, text="Mi correo: 4751083696n@gmail.com", fg="white", bg="#030075",
              font=("Arial", 10, "bold"), padx=5, pady=5).grid(row=2, column=0, columnspan=2, pady=5)

        Label(self.ventana, text="Destinatario:", fg="white", background="#030075", font=("Arial", 10, "bold"), padx=5, pady=5).grid(row=3, column=0)
        Entry(self.ventana, textvariable=self.destinatario, width=34).grid(row=3, column=1)

        Label(self.ventana, text="Asunto:", fg="white", background="#030075", font=("Arial", 10, "bold"), padx=5, pady=5).grid(row=4, column=0)
        Entry(self.ventana, textvariable=self.asunto, width=34).grid(row=4, column=1)

        Label(self.ventana, text="Mensaje:", fg="white", background="#030075", font=("Arial", 10, "bold"), padx=5, pady=5).grid(row=5, column=0)
        self.mensaje = Text(self.ventana, height=5, width=28, padx=5, pady=5)
        self.mensaje.grid(row=5, column=1)

        self.archivo_label = Label(self.ventana, text="Ningún archivo seleccionado", fg="gray", font=("Arial", 9))
        self.archivo_label.grid(row=6, column=0, columnspan=2)

        Button(self.ventana, text="SELECCIONAR ARCHIVO", command=self.seleccionar_archivo,
               height=2, width=20, bg="gray", fg="white", font=("Arial", 9)).grid(row=7, column=0, columnspan=2, padx=5, pady=5)

        Button(self.ventana, text="ENVIAR", command=self.enviar_email,
               height=2, width=10, bg="#FFC40A", fg="black", font=("Arial", 10, "bold")).grid(row=8, column=0, columnspan=2, padx=5, pady=10)

    def seleccionar_archivo(self):
        self.archivo_adjunto = filedialog.askopenfilename()
        if self.archivo_adjunto:
            nombre_archivo = os.path.basename(self.archivo_adjunto)
            self.archivo_label.config(text=f"📎 Archivo: {nombre_archivo}", fg="green")

    def enviar_email(self):
        remitente = "4751083696n@gmail.com"
        contrasena = "zras kris sqga zuob"

        email = EmailMessage()
        email["From"] = remitente
        email["To"] = self.destinatario.get()
        email["Subject"] = self.asunto.get()
        email.set_content(self.mensaje.get("1.0", "end"))

        if self.archivo_adjunto:
            mime_type, _ = mimetypes.guess_type(self.archivo_adjunto)
            if mime_type:
                maintype, subtype = mime_type.split('/')
            else:
                maintype, subtype = 'application', 'octet-stream'
            with open(self.archivo_adjunto, 'rb') as f:
                email.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                     filename=os.path.basename(self.archivo_adjunto))

        try:
            smtp = smtplib.SMTP_SSL("smtp.gmail.com")
            smtp.login(remitente, contrasena)
            smtp.send_message(email)
            smtp.quit()
            messagebox.showinfo("", "Mensaje enviado correctamente.")
        except Exception as e:
            messagebox.showerror("ERROR", f"No se pudo enviar el correo:\n{str(e)}")
