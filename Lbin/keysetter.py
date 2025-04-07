import keyring

# Replace these with your real credentials
broker_name = "xerces_meta"  # or "xerces_icm"
#username = "51698985"  # your MT5 account login
#password = "lsor31tz$r8aih"

#keyring.set_password(broker_name, username, password)

cred = keyring.get_credential("xerces_meta", "")
print(cred.username, cred.password)
print(f"User: {cred.username}")
print(f"Pass: {cred.password}")