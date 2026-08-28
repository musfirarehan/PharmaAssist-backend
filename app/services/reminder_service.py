def generate_reminders(medicines):

    reminders = []


    for medicine in medicines:

        name = medicine.get(
            "name",
            "Unknown"
        )

        dosage = medicine.get(
            "dosage",
            ""
        )

        frequency = medicine.get(
            "frequency",
            ""
        )

        duration = medicine.get(
            "duration",
            ""
        )


        # Default timing logic
        if "Once" in frequency:
            times = [
                "08:00 AM"
            ]

        elif "Twice" in frequency or "BD" in frequency:
            times = [
                "08:00 AM",
                "08:00 PM"
            ]

        elif "Three" in frequency or "TDS" in frequency:
            times = [
                "08:00 AM",
                "02:00 PM",
                "08:00 PM"
            ]

        elif "Four" in frequency or "QID" in frequency:
            times = [
                "08:00 AM",
                "12:00 PM",
                "04:00 PM",
                "08:00 PM"
            ]

        else:
            times = [
                "Follow doctor's instructions"
            ]


        reminders.append(
            {
                "medicine": name,
                "dosage": dosage,
                "times": times,
                "duration": duration
            }
        )


    return {
        "reminders": reminders
    }