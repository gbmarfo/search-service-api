from pydantic import BaseModel

class OrganizationBase(BaseModel):
    name: str | None = None
    description: str | None = None
    contact_name: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    contact_address: str | None = None
    organization_id: str | None = None
    organization_type: str | None = None

class OrganizationCreate(OrganizationBase):
    pass

class Organization(OrganizationBase):
    id: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    full_name: str | None = None
    username: str | None = None
    password: str | None = None
    email: str | None = None
    organization_id: str | None = None
    role: str | None = None
    is_active: int | None = None
    user_id: str | None = None

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int

    class Config:
        from_attributes = True

class SearchIndexBase(BaseModel):
    global_id: str | None = None
    title: str | None = None
    description: str | None = None
    table_name: str | None = None
    text_columns: str | None = None
    id_col: str | None = None
    org_id: str | None = None
    source: str | None = None
    schema_name: str | None = None

class SearchIndexCreate(SearchIndexBase):
    pass

class SearchIndex(SearchIndexBase):
    id: int
    created_by: str | None = None

    class Config:
        from_attributes = True


class IndexDocumentBase(BaseModel):
    global_id: str 
    index_id: str
    filename: str
    filepath: str

class IndexDocumentCreate(IndexDocumentBase):
    pass

class IndexDocument(IndexDocumentBase):
    id: int

    class Config:
        from_attributes = True