import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self, predictors):
        super().__init__()
        self.predictors = predictors

        self.num_preds = 3

        self.build_ui()

    
    def build_ui(self):
        self.geometry("700x500")
        self.title("Word Predictor")

        self.grid_columnconfigure((0, 1, 2), weight=1)

        self.text_entry = ctk.CTkEntry(self, placeholder_text="Skriv här...")
        self.text_entry.grid(row=0, column=0, 
                        columnspan=3, padx=20, pady=20,
                        sticky="ew")
        self.text_entry.bind("<KeyRelease>", self.on_key_release)

        self.model_selection = ctk.CTkOptionMenu(self, values=["Ngram", "Transformer"])
        self.model_selection.grid(row=1, column=0,
                                  padx=20, pady=20,
                                  sticky="ew")

        self.num_preds_label = ctk.CTkLabel(self, text="Number of predictions: ")
        self.num_preds_label.grid(row=1, column=1, 
                                  padx=20, pady=20,
                                  sticky="ew")

        num_preds_frame = ctk.CTkFrame(self, fg_color="transparent")
        num_preds_frame.grid(row=1, column=2, padx=20, pady=20, sticky="ew")
        num_preds_frame.grid_columnconfigure(0, weight=1)

        self.num_preds_entry = ctk.CTkEntry(num_preds_frame, placeholder_text="3")
        self.num_preds_entry.grid(row=0, column=0, 
                                  padx=(0, 5), sticky="ew")

        self.apply_button = ctk.CTkButton(num_preds_frame, text="Apply", width=60, command=self.apply_num_preds)
        self.apply_button.grid(row=0, column=1)

        self.suggestion_label = ctk.CTkLabel(self, text="Word predictions: ")
        self.suggestion_label.grid(row=2, column=0, 
                                  padx=20, pady=0,
                                  sticky="ew")

        self.suggestions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.suggestions_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

        self.spelling_label = ctk.CTkLabel(self, text="Spell checking: ")
        self.spelling_label.grid(row=4, column=0, 
                                  padx=20, pady=0,
                                  sticky="ew")

        self.spelling_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.spelling_frame.grid(row=5, column=0, columnspan=3, padx=20, pady=20, sticky="ew")


        self.suggestion_buttons = []
        self.spelling_buttons = []
        self.rebuild_suggestion_buttons()
        self.rebuild_spelling_buttons()

    def apply_num_preds(self):
        val = self.num_preds_entry.get()
        if val.isdigit() and int(val) > 0:
            self.num_preds = int(val)
            self.rebuild_suggestion_buttons()
            self.rebuild_spelling_buttons()

    def rebuild_suggestion_buttons(self):
        for btn in self.suggestion_buttons:
            btn.destroy()
        self.suggestion_buttons = []
        for i in range(self.num_preds):
            self.suggestions_frame.grid_columnconfigure(i, weight=1)
            btn = ctk.CTkButton(self.suggestions_frame, text="", command=lambda i=i: self.on_suggestion_click(i, "suggestion"))
            btn.grid(row=0, column=i, padx=5, sticky="ew")
            self.suggestion_buttons.append(btn)

    def rebuild_spelling_buttons(self):    
        for btn in self.spelling_buttons:
            btn.destroy()
        self.spelling_buttons = []
        for i in range(self.num_preds):
            self.spelling_frame.grid_columnconfigure(i, weight=1)
            btn = ctk.CTkButton(self.spelling_frame, text="", command=lambda i=i: self.on_suggestion_click(i, "spelling"))
            btn.grid(row=0, column=i, padx=5, sticky="ew")
            self.spelling_buttons.append(btn)

    def on_key_release(self, event):
        text = self.text_entry.get()
        model = self.model_selection.get()
        predictor = self.predictors[model.lower()]
        num_preds = 3 if self.num_preds_entry.get() == "" else int(self.num_preds_entry.get())

        suggestions = predictor.predict(text, num_preds)
        self.update_suggestion_btns(suggestions)
        self.update_spelling_btns(["spelling1", "spelling2"])       # TODO: get spelling suggesitons

    def on_suggestion_click(self, index, type):
        suggestion = None
        if type == "suggestion":
            suggestion = self.suggestion_buttons[index].cget("text")
        elif type == "spelling":
            suggestion = self.spelling_buttons[index].cget("text")
        else:
            print("Error: non-existent suggestion click type")
        current_text = self.text_entry.get()

        if suggestion == "":
            return

        if current_text.endswith(" "):
            new_text = f"{current_text}{suggestion} "
        else:
            words = current_text.rsplit(" ", 1)
            new_text = f"{words[0]} {suggestion} " if len(words) > 1 else suggestion + " "

        self.text_entry.delete(0, ctk.END)
        self.text_entry.insert(0, new_text)

        self.on_key_release(None)

    def update_suggestion_btns(self, suggestions):
        for i, btn in enumerate(self.suggestion_buttons):
            suggestion = suggestions[i] if len(suggestions) > i else ""
            btn.configure(text=suggestion)
    
    def update_spelling_btns(self, suggestions):
        for i, btn in enumerate(self.spelling_buttons):
            suggestion = suggestions[i] if len(suggestions) > i else ""
            btn.configure(text=suggestion)

    def run(self):
        self.mainloop()

