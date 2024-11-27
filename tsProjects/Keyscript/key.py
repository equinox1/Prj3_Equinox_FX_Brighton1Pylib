
import keyring as kr
kr.set_password("xerces_icm", "51698985", r"LsOR31tz$r8AIH") 
kr.set_password("xerces_meta", "5031861048", r"JyMj!o5g") 

kr.get_password("xerces_icm", "51698985")
kr.get_password("xerces_meta", "5031861048")
print(kr.get_password("xerces_icm", "51698985"))
print(kr.get_password("xerces_meta", "5031861048"))