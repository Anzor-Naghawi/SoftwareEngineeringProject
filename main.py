import customtkinter as tk
import io
from model import Model, tb
import PyPDF3 as pypdf3
import PyPDF2 as pypdf2


class App(tk.CTk):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.metadata = {}
        self.file_path = ""
        self.title("File Uploader")
        self.geometry(f"{300}x{200}")
        self.browse_button = tk.CTkButton(self, text="Browse", command=self.browse_file)
        self.browse_button.pack()
        self.label = tk.CTkLabel(self, text="No file selected")
        self.label.pack()
        self.scan_button = tk.CTkButton(self, text="Scan", command=self.extractmetadata)
        self.scan_button.pack()

    def browse_file(self):
        self.file_path = tk.filedialog.askopenfilename()
        self.label.configure(text=self.file_path)

    def extractmetadata(self):
        with open(self.file_path, 'rb') as file:
            pdf = pypdf3.PdfFileReader(file)
            pdf2 = pypdf2.PdfReader(file)
            metadata = pdf.getDocumentInfo()
            self.metadata["FileName"] = [self.file_path[self.file_path.rindex('/'):]]
            self.metadata["PdfSize"] = [pdf.__sizeof__()]
            self.metadata["MetadataSize"] = [metadata.__sizeof__()]
            self.metadata["Pages"] = [pdf.numPages]
            self.metadata["XrefLength"] = [len(pdf.xref)]
            self.metadata["TitleCharacters"] = [len(metadata["/Title"])]
            if pdf.isEncrypted:
                self.metadata["isEncrypted"] = [1]
            else:
                self.metadata["isEncrypted"] = [0]
            cat = pdf2.trailer["/Root"]
            self.metadata["EmbeddedFiles"] = [0]
            if "/Names" in cat:
                if "/EmbeddedFiles" in cat["/Names"]:
                    fn = cat["/Names"]['/EmbeddedFiles']['/Names']
                    self.metadata["EmbeddedFiles"] = [len(fn)]
            numimgs = 0
            for page in pdf2.pages:
                if page.get("/XObject") is not None:
                    numimgs += 1
            self.metadata["Images"] = [numimgs]
            for page in pdf2.pages:
                if page.extract_text() is not None:
                    self.metadata["Text"] = [1]
                    break
                else:
                    self.metadata["Text"] = [0]
            fl = pdf2.pdf_header
            self.metadata["Header"] = [f'%PDF-{fl[5:]}']
            self.metadata["Obj"] = [len(pdf.xref)-1]
            self.metadata["Endobj"] = [len(pdf.xref)-1]
            self.metadata["Stream"] = [len([obj for _, obj in pdf.resolvedObjects if obj.get('/Type') == '/Stream'])]
            self.metadata["Endstream"] = [0]
            self.metadata["Xref"] = [len(pdf.trailer)]
            self.metadata["Trailer"] = [len(pdf.trailer)]
            self.metadata["StartXref"] = [0]
            file.seek(-1024, io.SEEK_END)
            startxref = file.read().decode().lower().rfind('startxref')
            if startxref != -1:
                offset = file.tell() - 1024 + startxref
                file.seek(offset)
                startxrefl = file.readline().decode().strip()
                try:
                    space_index = startxrefl.rfind(' ')
                    offset = int(startxrefl[space_index+1:])
                except ValueError:
                    offset = 1
            self.metadata["StartXref"] = [offset]
            if pdf.numPages == 0:
                self.metadata["PageNo"] = [1]
            else:
                self.metadata["PageNo"] = [pdf.numPages]
            self.metadata["Encrypt"] = [0]
            self.metadata["ObjStm"] = [pdf.xref_objStm]
            self.metadata["JS"] = [0]
            self.metadata["Javascript"] = [0]
            self.metadata["AA"] = [0]
            self.metadata["OpenAction"] = [0]
            self.metadata["Acroform"] = [0]
            self.metadata["JBIG2Decode"] = [0]
            self.metadata["RichMedia"] = [0]
            self.metadata["Launch"] = [0]
            self.metadata["EmbeddedFile"] = [0]
            self.metadata["XFA"] = [0]
            self.metadata["Colors"] = [0]
            print(tb.tabulate(self.metadata, headers='keys', tablefmt='psql'))

    def scan(self):
        pass

    def logdata(self, tabulate, size):
        self.model.data.print(tabulate=tabulate, size=size)

if __name__ == "__main__":
    app = App()
    app.model.split()
    app.logdata(True, 1)
    app.model.scale()
    # app.model.plots()
    app.model.test()

    print("MODEL 2")
    app.model.fitmodel2()
    app.model.test2()

    app.mainloop()


