import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self, predictors):
        super().__init__()
        self.predictors = predictors

        self.build_ui()

    
    def build_ui(self):
        self.geometry("700x500")
        self.title("Word Predictor")

        self.text_entry = ctk.CTkEntry(self, placeholder_text="Skriv här...")
        self.text_entry.grid(row=1, column=0,
                        columnspan=3, padx=20, pady=20,
                        sticky="ew")
        self.text_entry.bind("<KeyRelease>", self.on_key_release)
        
        self.model_selection = ctk.CTkOptionMenu(self, values=["Ngram", "Transformer"])
        self.model_selection.grid(row=2, column=0,
                                  padx=20, pady=20,
                                  sticky="ew")
        
        self.num_preds_label = ctk.CTkLabel(self, text="Number of predictions: ")
        self.num_preds_label.grid(row=2, column=1,
                                  padx=20, pady=20,
                                  sticky="ew")
        
        self.num_preds_entry = ctk.CTkEntry(self, placeholder_text="3")
        self.num_preds_entry.grid(row=2, column=2,
                                  padx=20, pady=20,
                                  sticky="ew")
        
        self.suggestion_buttons = []
        for i in range(3):                          # TODO: fixa så att den kan variera antalet knappar för suggestions
            btn = ctk.CTkButton(self, text="", command= lambda i=i: self.on_suggestion_click(i))
            btn.grid(row=3, column=i,
                     padx=20, pady=20,
                     sticky="ew")
            self.suggestion_buttons.append(btn)


    def on_key_release(self, event):
        text = self.text_entry.get()
        model = self.model_selection.get()
        predictor = self.predictors[model.lower()]
        num_preds = 3 if self.num_preds_entry.get() == "" else int(self.num_preds_entry.get())

        suggestions = predictor.predict(text, num_preds)
        self.update_suggestion_btns(suggestions)

    def on_suggestion_click(self, index):
        suggestion = self.suggestion_buttons[index].cget("text")
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

    def run(self):
        self.mainloop()

